-- dialogue_ui.lua (PLAYER)
-- H near NPC -> lock NPC, send auto-greeting, open a modal chat panel with a
-- native OpenMW text field (supports Russian/any layout via engine text input).
-- Enter/F1 sends, Esc closes. While open, the game switches to Interface mode:
-- the mouse cursor is freed and player movement is disabled, so keystrokes go
-- to the text field instead of walking the character.
--
-- OpenMW 0.49+, Lua 5.1

local ui      = require('openmw.ui')
local camera  = require('openmw.camera')
local input   = require('openmw.input')
local self_   = require('openmw.self')
local nearby  = require('openmw.nearby')
local types   = require('openmw.types')
local util    = require('openmw.util')
local core    = require('openmw.core')
local async   = require('openmw.async')
local I       = require('openmw.interfaces')
local vfs     = require('openmw.vfs')
local json    = require('scripts.json')
local v2      = util.vector2

-- IPC files (relative to the mod's data= root, read/written by the Python bridge)
local RESPONSE_VFS = 'ai_inbox/response.txt'

local MAX_DIST  = 500
local HAIL_KEY  = input.KEY.H
local NARRATOR_KEY = input.KEY.U   -- talk to the Narrator (BG3-style voice-over)
local VOICE_KEY = input.KEY.V      -- voice mode: look at an NPC, speak, no windows
local SEND_KEY  = input.KEY.F1
local CLOSE_KEY = input.KEY.Escape

local narratorMode = false   -- window is a conversation with the Narrator
local voiceTalking = false   -- push-to-talk: the V key is currently held

local INTERFACE_MODE = (I.UI and I.UI.MODE and I.UI.MODE.Interface) or 'Interface'

-- ВНИМАНИЕ, ЖЁСТКИЙ ПРЕДЕЛ: у Lua не больше 200 локальных переменных в главном
-- блоке файла, и мы у самой границы. Двадцать две отдельные переменные под
-- сцены уже роняли скрипт целиком («main function has more than 200 local
-- variables»), а с ним переставали работать H и V. Поэтому всё, что относится
-- к сценам — числа, состояние и функции, — живёт в ОДНОЙ таблице. Объявлена
-- рано, чтобы её видели и функции окна, и обработчик кадра.
local SC = {
    RADIUS = 1200,     -- кого берём в состав
    CAST_MAX = 5,
    WALK_MAX = 12,     -- сколько ждём, пока дойдёт (сторож от зависания сцены)
    -- ЧАСТОТА СОБЫТИЙ. Считали вместе с игроком: при прежних числах верхняя
    -- граница выходила 58 событий в час — радиант 40, оклики 12, сцены 6, то
    -- есть по событию в минуту. Свели к ОДНОМУ В ПЯТЬ МИНУТ: 5 радиантов,
    -- 4 оклика и 3 сцены в час.
    --
    -- Снижен не только откат, но и ШАНС. При 10% сцена срабатывала почти сразу
    -- после отката, и события шли как по будильнику — ровно раз в 11 минут.
    -- При 2% ожидание после отката минут пять и с большим разбросом: мир
    -- перестаёт быть предсказуемым.
    COOLDOWN = 900,    -- между случайными сценами, секунды
    CHANCE = 0.02,     -- проверка раз в 5 с при подходящем составе
    -- Ручки характера мира, приезжают из ai_inbox/tuning.txt (см. настройки-мира.txt)
    danger = 30,
    humour = 30,
    ARRIVED = 220,     -- ближе этого считаем, что дошёл
    REPLY_W = 620, REPLY_H = 260,   -- размер окна разговора
    cur = nil,         -- идущая сцена: {beats, i, phase, timer, actors}
    askedAt = -math.huge,
    lastAt = -math.huge,
    director = false,  -- окно открыто для указания режиссёра
}

local lockedCtx     = nil
local lockedNpcObj  = nil   -- GameObject of the locked NPC (for real actions)
local companionObj  = nil   -- NPC currently following the player (via ACTION:follow)
local companionCtx  = nil   -- their context snapshot (name/race/gender/...)
local companionLoss = nil   -- {name, until_t}: recent companion death, shown in scene
local lastReplyText = ''
local lastSpeaker   = ''    -- who said lastReplyText ('' = the locked NPC)
local lastEmotion   = ''
local inputBuffer   = ''
local window        = nil
local isOpen        = false

-- IPC state (all requests/replies flow through THIS player script so they work
-- while the game is paused in the chat window — onFrame runs during pause).
local reqCounter    = 0
local lastRespReqId = ''
local pollTimer     = 0
-- «Ждём ответа» без конца обычно значит, что моста нет: окно закрыли или он
-- упал. Молчащий NPC выглядит поломкой мода, поэтому считаем время ожидания.
local waitTimer     = -1     -- -1 = ничего не ждём
local waitWarned    = false
-- Session salt: the bridge outlives game sessions and dedups req_ids; without
-- a per-session random component ids repeat across reloads and the bridge
-- silently drops requests ('ждём ответа' forever).
local SESSION_SALT  = tostring(math.random(100000, 999999))

-- Per-NPC conversation history, stored IN THE SAVEGAME (onSave/onLoad below).
-- This is the authoritative memory: reloading an earlier save rolls the NPCs'
-- memories back with it, so they can't "remember the future".
local npcHistory = {}   -- npc_id -> { {role='player'|'npc', content=...}, ... }
-- Save-scoped too (same reasoning — reloading must rewind ALL of these):
local npcMood    = {}   -- npc_id -> last emotion toward the player
local npcFacts   = {}   -- npc_id -> {3 life facts} (generated once by the bridge)
local worldRumors = {}  -- {text, region, day} — news with a place and a date
local debts      = {}   -- npc_id -> {name=..., amount=..., due=<game day>}
local escort     = nil  -- {npc_id,name,cellKey,townRu,reward,dueDay} — live contract
local duel       = nil  -- {npc_id,name,stake,started} — live duel of honour
-- Who laid down arms and when (sim time). Read by the death watch far above
-- trySurrender, so it must be declared here — Lua locals are visible downward.
local surrenderedAt = {}

-- СУДЬБА. У большинства она не меняется никогда. Но если разговор к чему-то
-- привёл — человек уехал, завязал, спился, поднялся — жизнь идёт дальше и БЕЗ
-- игрока: судьба взрослеет по игровым дням, и при следующей встрече он уже
-- другой. Живёт в сейве: откат сохранения откатывает и судьбу.
-- npc_id -> {role=..., town=..., day0=<игровой день>, stage=..., owes=<долг игроку>}
local npcFate = {}

-- Партийная РПГ: у спутника есть своя скрытая история из трёх ступеней. Она
-- рождается один раз (мост придумывает её по канону персонажа) и открывается
-- не по времени, а по тому, как далеко продвинулся ИГРОК: главный квест,
-- линии фракций, ранги. Лежит в сейве вместе со всем остальным.
local npcArc = {}   -- npc_id -> {ступень1, ступень2, ступень3}

-- Насколько далеко зашла история игрока: 1 — только начал, 3 — глубоко внутри.
-- Считаем по главной линии (кодовые квесты a1_/a2_/b1_...) и числу взятых дел.
local function storyProgress()
    local level, main = 1, 0
    pcall(function()
        local total = 0
        for _, q in ipairs(types.Player.quests(self_.object)) do
            total = total + 1
            local id = tostring(q.id or ''):lower()
            local st = tonumber(q.stage) or 0
            if id:sub(1, 2) == 'a2' or id:sub(1, 2) == 'b2' or id:sub(1, 2) == 'c3' then
                main = math.max(main, 2)
            elseif id:sub(1, 2) == 'a1' or id:sub(1, 2) == 'b1' then
                main = math.max(main, 1)
            end
            if id:sub(1, 2) == 'a2' and st >= 30 then main = 3 end
        end
        level = math.max(1, math.min(3, math.max(main, math.floor(total / 8) + 1)))
    end)
    return level
end

local FATE_STORY = {
    worker   = { 'осваивается на новом месте, руки в мозолях',
                 'зацепился за работу, платят скудно, но честно',
                 'встал на ноги — своё дело, своя копейка' },
    innkeep  = { 'подносит кружки за угол и еду',
                 'стал своим при таверне, знает все слухи',
                 'откладывает на собственное заведение' },
    drunk    = { 'первые дни держался, потом сорвался',
                 'пропил всё до нитки',
                 'опустился совсем, ночует где придётся' },
    beggar   = { 'сидит с протянутой рукой у ворот',
                 'знает каждую подворотню, живёт подаянием',
                 'прибился к нищей братии, там свои порядки' },
    guard    = { 'взяли в младшие стражники, гоняют как новичка',
                 'несёт службу, начал получать жалованье',
                 'на хорошем счету у начальства' },
    smuggler = { 'снова взялся за старое, но осторожнее',
                 'ходит с грузом по ночам, при деньгах',
                 'поднялся в деле, и это до добра не доведёт' },

    -- ── ЗАЦИКЛЕННАЯ ─────────────────────────────────────────────────────
    -- Судьба, которая не кончается, а возвращается к началу. Человек уезжает
    -- в город мечты, там не складывается — и он снова копит на билет, только
    -- уже в следующий город. Так его можно прогнать по всей провинции, пока
    -- он не вернётся туда, откуда начал, и не начнёт заново.
    -- Шутка тут длинная: игрок про него забывает и встречает через полдня
    -- игры совсем в другом конце Вварденфелла — за тем же самым.
    ticket   = { loop = true,
                 'обжился было, но всё оказалось не так, как обещали',
                 'перебивается кое-как и уже поглядывает на дорогу',
                 'снова сидит с протянутой рукой и копит на билет — '
                 .. 'теперь-то уж точно в правильный город' },

    -- ── НЕЛЕПЫЕ ─────────────────────────────────────────────────────────
    -- Судьба нелепа сама по себе, а человек проживает её на полном серьёзе.
    actor    = { 'прибился к бродячим комедиантам таскать сундуки',
                 'вышел на сцену — играет заднюю ногу гуара',
                 'играет заднюю ногу гуара так, что и на улице его зовут Ногой' },
    prophet  = { 'начал толковать прохожим свой вещий сон',
                 'обзавёлся двумя слушателями и чужой мантией не по размеру',
                 'основал учение; последователей по-прежнему двое, и оба должны ему денег' },
    fisher   = { 'поклялся поймать ту самую рыбу, что сорвалась',
                 'просиживает на мостках с рассвета, рыбы не видит',
                 'о рыбе больше не говорит, но с мостков не уходит' },
    clerk    = { 'устроился переписчиком в контору',
                 'нашёл в бумагах ошибку двадцатилетней давности',
                 'ведёт с конторой переписку об этой ошибке; работать некогда' },
    guard_ic = { 'взяли в стражу, дали пост у пустого сарая',
                 'охраняет сарай образцово, внутрь так и не заглядывал',
                 'сарай снесли, пост остался — стоит и охраняет место' },

    -- ── НА МЕСТЕ ────────────────────────────────────────────────────────
    -- Человек никуда не уезжает: остаётся при своём доме, своих репликах и
    -- своём задании. Меняется только его жизнь — и то, как он о ней говорит.
    -- Это единственные судьбы, доступные квестовым: увозить их нельзя, а
    -- новая глава в жизни квесту не мешает.
    hoarder  = { 'уверился, что прятать надёжнее, чем носить при себе',
                 'завёл тайники по всей округе и путается в них',
                 'ищет собственные схроны и подозревает в пропажах соседей' },
    devotee  = { 'рассказывает всем встречным, как ему помогли',
                 'история обросла подробностями, которых не было',
                 'в его рассказе спаситель уже воевал с даэдра в одиночку' },
    lucky    = { 'решил, что с того дня ему везёт',
                 'испытывает удачу по мелочи и всякий раз убеждается в своём',
                 'ставит на кон всё подряд; удача пока держится, и это пугает' },
    sleuth   = { 'раз пропажа нашлась, объявил себя знатоком пропаж',
                 'берётся искать чужое и находит совсем не то',
                 'ведёт учёт раскрытых дел; в списке одно, и то чужое' },
    keeper   = { 'решил отплатить и присматривает за благодетелем издали',
                 'заводит на него подробную опись: что ел, куда ходил',
                 'считает себя ответственным за его судьбу и очень серьёзен' },
}
-- Через сколько игровых дней судьба переходит на следующую ступень.
local FATE_STEP_DAYS = 3

local function fateNote(npcId)
    local f = npcFate[tostring(npcId or '')]
    if not f then return '' end
    local line = FATE_STORY[f.role]
    if not line then return '' end
    local day = math.floor((core.getGameTime() or 0) / 86400)
    local elapsed = math.max(0, day - (f.day0 or day))
    local stage = math.min(#line, 1 + math.floor(elapsed / FATE_STEP_DAYS))
    f.stage = stage
    local s
    if f.stay then
        -- Никуда не уезжал: то же место, та же работа, то же задание. Новая
        -- глава появилась только в жизни.
        s = 'ЧТО С ТОБОЙ СТАЛО ПОСЛЕ ТОГО РАЗГОВОРА (ты всё там же, при своих ' ..
            'делах; прошло дней: ' .. elapsed .. '): ' .. line[stage] .. '.'
        if f.because and f.because ~= '' then
            s = s .. ' Началось всё с того, что игрок тогда сказал и сделал: «' ..
                f.because .. '». ТА БЕДА ПОЗАДИ — больше на неё не жалуйся и не ' ..
                'ищи того, что уже нашлось. Даже если твои привычные слова о ней ' ..
                'ещё звучат в округе, для ТЕБЯ вопрос закрыт.'
        end
    else
        s = 'ТВОЯ СУДЬБА ПОСЛЕ ТОГО РАЗГОВОРА: ты в городе ' ..
            tostring(f.town or '?') .. ', прошло дней: ' .. elapsed ..
            '. Сейчас ты ' .. line[stage] .. '.'
    end
    -- Без этого NPC отмахивался одной грубостью («какая тебе разница»), и вся
    -- судьба оставалась за кадром: игрок не узнавал ни где человек, ни кем
    -- стал. Ругаться никто не запрещает — но СНАЧАЛА по существу.
    s = s .. ' ОТВЕЧАЙ ПО СУЩЕСТВУ: скажи прямо, чем ты теперь занят и как ' ..
        'дошёл до такого. НЕ НАЧИНАЙ с отговорки и встречного вопроса ' ..
        '(«какая тебе разница», «тебе-то что») — про свою жизнь ты ' ..
        'рассказываешь охотно, с первого же слова и по делу. Говори как о ' ..
        'деле, а не как о курьёзе: смешным своё положение ты не считаешь. ' ..
        'И держись СВОЕЙ судьбы — чужих занятий себе не приписывай.'
    -- Зацикленная судьба: на последней ступени человек снова просит на билет.
    -- Города, где он уже побывал, называть новой целью нельзя — иначе круг
    -- выродится в топтание между двумя, и длинная шутка сломается.
    if line.loop and stage >= #line then
        local been = {}
        for _, t in ipairs(f.towns or {}) do been[#been + 1] = t end
        if f.town and f.town ~= '' then been[#been + 1] = f.town end
        s = s .. ' ТЫ СНОВА КОПИШЬ НА БИЛЕТ и просишь у собеседника денег на ' ..
            'дорогу — ровно так же, как просил в прошлый раз, и совершенно ' ..
            'серьёзно: вот на этот раз всё точно сложится, потому что <назови ' ..
            'свою причину, каждый раз новую и убедительную>. ' ..
            'КУДА именно — выбери ОДИН город Вварденфелла из этих: Балмора, ' ..
            'Вивек, Альд\'рун, Кальдера, Пелагиад, Садрит Мора, Гнисис, Суран, ' ..
            'Эбенгард, Кхуул, Маар Ган, Гнаар Мок, Вос, Хла Оуд. ' ..
            'НЕЛЬЗЯ называть те, где ты уже пожил и не сложилось: ' ..
            table.concat(been, ', ') .. '. Нельзя и Сиродил с Имперским ' ..
            'городом — на них у тебя денег не хватит никогда. ' ..
            'Если из списка не осталось ни одного — скажи, что понял главное, ' ..
            'и просишь на дорогу ДОМОЙ, туда, откуда всё начиналось. ' ..
            'Ты НЕ ПОМНИШЬ этой закономерности и не находишь в ней ничего ' ..
            'странного: каждый раз это первая и единственная твоя попытка.'
    end
    if (f.owes or 0) > 0 then
        s = s .. ' Ты остался должен игроку ' .. tostring(f.owes) ..
            ' золотых. Если дела пошли в гору — верни с лихвой, по совести; ' ..
            'если всё пропил — стыдись и оправдывайся.'
    end
    return s
end

-- Память двухслойная. Обычная болтовня живёт недолго и вытесняется, а строки,
-- помеченные как ФАКТ (долг, уговор, клятва, содеянное), не выбрасываются
-- никогда: у спутника после часа пути именно они пропадали первыми, и он
-- забывал, что ему спасли жизнь и о чём договаривались.
local HISTORY_MAX, FACTS_MAX = 36, 14

local function isFact(entry)
    return tostring(entry and entry.content or ''):find('(ФАКТ', 1, true) ~= nil
end

local function pushHistory(npcId, role, content)
    npcId = tostring(npcId or '')
    if npcId == '' then return end
    local h = npcHistory[npcId]
    if not h then h = {}; npcHistory[npcId] = h end
    h[#h + 1] = { role = role, content = string.sub(tostring(content or ''), 1, 240) }

    while #h > HISTORY_MAX do
        -- Сначала выбрасываем самую старую ОБЫЧНУЮ реплику.
        local dropped = false
        for i = 1, #h do
            if not isFact(h[i]) then table.remove(h, i); dropped = true; break end
        end
        -- Если остались одни факты — жертвуем самым старым из них.
        if not dropped then table.remove(h, 1) end
    end
end

-- Модели уходят ДВА слоя: все запомненные факты (даже давние) и свежий кусок
-- разговора. Раньше отправлялись просто последние 20 строк, и факт двадцать
-- пятой давности был для NPC невидим, хотя лежал в памяти.
local function recentHistory(npcId)
    local h = npcHistory[tostring(npcId or '')] or {}
    local facts, recent = {}, {}
    local firstRecent = math.max(1, #h - 13)
    for i = 1, #h do
        if i >= firstRecent then
            recent[#recent + 1] = h[i]
        elseif isFact(h[i]) and #facts < FACTS_MAX then
            facts[#facts + 1] = h[i]
        end
    end
    local out = {}
    for _, e in ipairs(facts) do out[#out + 1] = e end
    for _, e in ipairs(recent) do out[#out + 1] = e end
    return out
end

-- Rumors carry WHERE and WHEN they were born. News travels at the speed of
-- caravans: in another region it only surfaces days later, and the further it
-- travelled the vaguer it gets (the prompt is told how second-hand it is).
local function addRumor(text)
    text = tostring(text or '')
    if text == '' then return end
    local last = worldRumors[#worldRumors]
    if last and (type(last) == 'table' and last.text or last) == text then return end
    local region, day = '', 0
    pcall(function()
        local cell = self_.object.cell
        if cell and cell.region then region = tostring(cell.region) end
        day = math.floor((core.getGameTime() or 0) / 86400)
    end)
    worldRumors[#worldRumors + 1] = { text = text, region = region, day = day }
    while #worldRumors > 20 do table.remove(worldRumors, 1) end
end

-- Which rumors have plausibly reached THIS place by now, annotated with how
-- distorted they are on arrival.
local function rumorsHere()
    local region, day = '', 0
    pcall(function()
        local cell = self_.object.cell
        if cell and cell.region then region = tostring(cell.region) end
        day = math.floor((core.getGameTime() or 0) / 86400)
    end)
    local out = {}
    for i = #worldRumors, 1, -1 do
        if #out >= 3 then break end
        local r = worldRumors[i]
        if type(r) ~= 'table' then
            out[#out + 1] = tostring(r)          -- legacy plain-string rumor
        else
            local age = day - (r.day or 0)
            if r.region ~= '' and region ~= '' and r.region ~= region then
                -- foreign news: needs ~2 days to travel, and arrives blurred
                if age >= 2 then
                    out[#out + 1] = r.text ..
                        ' [дошло из чужих краёв через ' .. age .. ' дн. — детали смутны, имена могли переврать]'
                end
            else
                if age >= 1 then
                    out[#out + 1] = r.text .. ' [местная молва, ' .. age .. ' дн. назад]'
                else
                    out[#out + 1] = r.text .. ' [свежая местная новость, ещё горячая]'
                end
            end
        end
    end
    return out
end

-- Widget references we mutate in place (so we don't rebuild & lose focus).
local replyWidget   = nil
local inputWidget   = nil

-- Forward declaration so the layout's close button can reference it.
local closeWindow

-- ── Helpers ───────────────────────────────────────────────────────────────────

local function showMsg(text)
    pcall(function() ui.showMessage(tostring(text or '')) end)
end

local EMOTION_COLORS = {
    neutral   = util.color.rgb(0.90, 0.90, 0.90),
    happy     = util.color.rgb(0.60, 1.00, 0.60),
    angry     = util.color.rgb(1.00, 0.40, 0.40),
    fearful   = util.color.rgb(0.80, 0.70, 1.00),
    disgusted = util.color.rgb(0.70, 0.90, 0.40),
    surprised = util.color.rgb(1.00, 0.90, 0.40),
}
local function replyColor()
    return EMOTION_COLORS[lastEmotion] or EMOTION_COLORS.neutral
end

-- Combat awareness: the world must know a fight is happening. We watch the
-- player's health for real damage and look for drawn steel nearby, so nobody
-- can chat politely while blades are out.
local lastHitAt  = -math.huge
local lastHealth = nil

local function combatWatch()
    pcall(function()
        local h = types.Actor.stats.dynamic.health(self_.object)
        if lastHealth and h.current < lastHealth - 0.5 then
            lastHitAt = core.getSimulationTime()
        end
        lastHealth = h.current
    end)
end

-- Recent killings the player was around for: who died, when, and by whose hand.
-- { [recordId] = {name=..., at=simTime, killer='', cell=''} }
local recentKills = {}

-- Bodies lying in plain sight. Every other scene helper skips the dead, so a
-- corpse at an NPC's feet was invisible to them: a guard killed a man on the
-- player's request, then went on asking where the culprit had got to.
local function corpsesNote()
    local out = {}
    pcall(function()
        for _, act in ipairs(nearby.actors or {}) do
            if act ~= self_.object and act.type == types.NPC and #out < 3 then
                local dead = false
                pcall(function() dead = types.Actor.isDead(act) end)
                if dead then
                    local d = (act.position - self_.object.position):length()
                    if d < 900 then
                        local id = tostring(act.recordId or '')
                        local nm = id
                        pcall(function() nm = types.NPC.record(act).name or id end)
                        local info = recentKills[id]
                        local s = 'тело: ' .. nm
                        if info and info.killer ~= '' then
                            s = s .. ' (убит(а) — ' .. info.killer .. ')'
                        end
                        if info then
                            local mins = math.floor((core.getSimulationTime() - info.at) / 60)
                            s = s .. (mins <= 1 and ', только что' or
                                      (', ' .. mins .. ' мин назад'))
                        end
                        out[#out + 1] = s
                    end
                end
            end
        end
    end)
    if #out == 0 then return '' end
    return 'РЯДОМ ЛЕЖАТ МЁРТВЫЕ — ' .. table.concat(out, '; ') ..
           '. Ты это видишь и не можешь делать вид, что ничего не произошло.'
end

-- Who, of the armed people standing over the body, most likely struck it down.
-- Returns the display name AND the actor, because the killer has to REMEMBER
-- doing it: a guard who had just run a man through went on asking the player
-- where the culprit might be hiding.
local function guessKiller(victim)
    local best, bestObj, bestD = '', nil, 1e9
    pcall(function()
        for _, act in ipairs(nearby.actors or {}) do
            local alive = false
            pcall(function() alive = not types.Actor.isDead(act) end)
            if alive and act ~= victim then
                local d = (act.position - victim.position):length()
                if d < 400 and types.Actor.getStance(act) == types.Actor.STANCE.Weapon then
                    local nm = ''
                    if act == self_.object then
                        nm = 'ты сам(а)'
                    elseif act.type == types.NPC then
                        pcall(function() nm = tostring(types.NPC.record(act).name or '') end)
                        if companionObj and act == companionObj then
                            nm = nm ~= '' and (nm .. ', твой спутник') or 'твой спутник'
                        end
                    end
                    if nm ~= '' and d < bestD then best, bestObj, bestD = nm, act, d end
                end
            end
        end
    end)
    return best, bestObj
end

-- A scene line about the fight, or '' when all is calm.
local function combatNote()
    local now = core.getSimulationTime()
    local hurtRecently = (now - lastHitAt) < 12
    local armed = {}
    pcall(function()
        for _, act in ipairs(nearby.actors or {}) do
            if act ~= self_.object and #armed < 3 and not types.Actor.isDead(act) then
                local d = (act.position - self_.object.position):length()
                if d < 700 and types.Actor.getStance(act) == types.Actor.STANCE.Weapon then
                    local nm = ''
                    pcall(function()
                        if act.type == types.NPC then
                            nm = tostring(types.NPC.record(act).name or '')
                        else
                            nm = tostring(act.recordId or '')
                        end
                    end)
                    armed[#armed + 1] = (nm ~= '' and nm) or 'кто-то'
                end
            end
        end
    end)
    if #armed == 0 and not hurtRecently then return '' end
    local s = 'ПРЯМО СЕЙЧАС ИДЁТ БОЙ'
    if #armed > 0 then s = s .. ': с оружием наголо рядом — ' .. table.concat(armed, ', ') end
    if hurtRecently then s = s .. '; игрока только что ранили' end
    return s .. ' — говорить учтиво посреди этого нелепо: кричи, отбивайся, требуй помощи или беги'
end

-- Whose roof is this? Inside an interior we count the owners of the objects
-- lying around: the dominant owner IS the householder. Real engine data, so
-- "you are standing in my house at 3am" is a fact, not a guess.
local function interiorOwner()
    local best, bestN = nil, 0
    pcall(function()
        local cell = self_.object.cell
        if not cell or cell.isExterior then return end
        local tally = {}
        local function tallyOwner(o)
            local id = o and o.owner and o.owner.recordId
            if id and id ~= '' then
                id = tostring(id):lower()
                tally[id] = (tally[id] or 0) + 1
            end
        end
        for _, it in ipairs(nearby.items or {}) do tallyOwner(it) end
        for _, c in ipairs(nearby.containers or {}) do tallyOwner(c) end
        for _, d in ipairs(nearby.doors or {}) do tallyOwner(d) end
        for id, n in pairs(tally) do
            if n > bestN then best, bestN = id, n end
        end
        if bestN < 2 then best = nil end   -- one stray item proves nothing
    end)
    return best
end

-- Where the NPC is standing, TOLD FROM THEIR SIDE. Without this a guard who
-- had followed the player into a stranger's house insisted he was at home.
-- (Must sit AFTER interiorOwner: in Lua a call to a local declared further
-- down silently reads nil and kills the whole handler.)
local function npcPlaceNote(npcObj)
    local note = ''
    pcall(function()
        local cell = self_.object.cell
        if not cell then return end
        if cell.isExterior then
            note = 'Вы оба под открытым небом' ..
                   (cell.name ~= '' and (', ' .. tostring(cell.name)) or '') .. '.'
            return
        end
        local ownerId = interiorOwner()
        local mine = ''
        pcall(function() mine = tostring(npcObj.recordId or ''):lower() end)
        if not ownerId then
            note = 'Вы внутри помещения' ..
                   (cell.name ~= '' and (' — ' .. tostring(cell.name)) or '') ..
                   '. Оно тебе не принадлежит.'
            return
        end
        if ownerId == mine then
            note = 'Ты у себя дома, и игрок пришёл к тебе.'
            return
        end
        -- Put a name to the owner: they may be standing here, or lying here.
        local ownerName = ''
        pcall(function()
            for _, a in ipairs(nearby.actors or {}) do
                if tostring(a.recordId or ''):lower() == ownerId and a.type == types.NPC then
                    ownerName = tostring(types.NPC.record(a).name or '')
                    break
                end
            end
        end)
        if ownerName == '' and recentKills[ownerId] then
            ownerName = recentKills[ownerId].name
        end
        note = 'ТЫ НАХОДИШЬСЯ В ЧУЖОМ ЖИЛИЩЕ' ..
               (ownerName ~= '' and (' — это дом, где живёт ' .. ownerName) or '') ..
               '. Это НЕ твой дом, ты зашёл сюда вместе с игроком.'
    end)
    return note
end

-- Can this actor actually SEE the player right now? (eye-height ray, so a wall
-- or a closed door means they saw nothing.)
local function hasLineOfSight(act)
    local seen = false
    pcall(function()
        local eye = act.position + util.vector3(0, 0, 120)
        local tgt = self_.object.position + util.vector3(0, 0, 90)
        local res = nearby.castRay(eye, tgt, { ignore = act })
        seen = (not res) or (not res.hit) or (res.hitObject == self_.object)
    end)
    return seen
end

-- ── NPC context builder ───────────────────────────────────────────────────────

local function buildNpcContext(npcObj)
    local ctx = { npc_id='', npc_record='', npc_name='', npc_race='', npc_class='', npc_faction='', location='', npc_is_male=true }
    -- npc_id = UNIQUE instance id: two different "Стражник" guards are two
    -- different people (separate memory, voice, facts). npc_record = ESM
    -- record id, used for canonical dialogue lookup.
    pcall(function() ctx.npc_id = tostring(npcObj.id or npcObj.recordId or '') end)
    pcall(function() ctx.npc_record = tostring(npcObj.recordId or '') end)
    pcall(function()
        local rec = types.NPC.record(npcObj)
        if rec then
            ctx.npc_name    = tostring(rec.name    or '')
            ctx.npc_race    = tostring(rec.race    or '')
            ctx.npc_class   = tostring(rec.class   or '')
            ctx.npc_faction = tostring(rec.faction or '')
            if rec.isMale ~= nil then ctx.npc_is_male = rec.isMale and true or false end
        end
    end)
    pcall(function()
        if self_.object and self_.object.cell then
            ctx.location = tostring(self_.object.cell.name or '')
        end
    end)
    return ctx
end

-- Find a live actor by its unique object id (for delayed replies whose
-- original reference may have gone stale).
local function findActorById(aid)
    aid = tostring(aid or '')
    if aid == '' then return nil end
    for _, act in ipairs(nearby.actors or {}) do
        local hit = false
        pcall(function() hit = tostring(act.id) == aid and not types.Actor.isDead(act) end)
        if hit then return act end
    end
    return nil
end

-- Find someone nearby by the name the model used: a plot against a third party
-- is only real if the engine can point at the person being plotted against.
local function findActorByName(name)
    local want = tostring(name or ''):lower()
    if want == '' or want == 'none' or want == 'player' or want == 'игрок' then return nil end
    local best, bestD = nil, 1e9
    pcall(function()
        for _, act in ipairs(nearby.actors or {}) do
            if act ~= self_.object and act.type == types.NPC
               and not types.Actor.isDead(act) then
                local nm = tostring(types.NPC.record(act).name or ''):lower()
                if nm ~= '' and (nm == want or nm:find(want, 1, true)
                                 or want:find(nm, 1, true)) then
                    local d = (act.position - self_.object.position):length()
                    if d < bestD then best, bestD = act, d end
                end
            end
        end
    end)
    return best
end

-- A door or container the player named ("иди к двери", "открой сундук").
local function findPlaceByName(name)
    local want = tostring(name or ''):lower()
    local best, bestD = nil, 1e9
    pcall(function()
        local lists = { nearby.doors or {}, nearby.containers or {} }
        for _, list in ipairs(lists) do
            for _, obj in ipairs(list) do
                local nm = ''
                pcall(function() nm = tostring(obj.type.record(obj).name or ''):lower() end)
                local d = (obj.position - self_.object.position):length()
                local match = (want == '' or want == 'none')
                    or (nm ~= '' and (nm:find(want, 1, true) or want:find(nm, 1, true)))
                if match and d < bestD then best, bestD = obj, d end
            end
        end
    end)
    return best
end

-- NPC the player is LOOKING AT: crosshair ray first, then a narrow view cone.
local function findAimedNpc()
    local hit = nil
    pcall(function()
        local origin = camera.getPosition()
        local dir = camera.viewportToWorldVector(util.vector2(0.5, 0.5))
        local res = nearby.castRay(origin, origin + dir * 600,
            { ignore = self_.object })
        if res and res.hitObject and res.hitObject.type == types.NPC
           and not types.Actor.isDead(res.hitObject) then
            hit = res.hitObject
        end
        if not hit then
            -- cone fallback: nearest NPC within ~14 degrees of the view axis
            local bestCos = 0.97
            for _, act in ipairs(nearby.actors or {}) do
                if act ~= self_.object and act.type == types.NPC
                   and not types.Actor.isDead(act) then
                    local v = (act.position + util.vector3(0, 0, 90)) - origin
                    local d = v:length()
                    if d > 1 and d <= 600 then
                        local c = v:normalize():dot(dir:normalize())
                        if c > bestCos then bestCos = c; hit = act end
                    end
                end
            end
        end
    end)
    return hit
end

local function findNearestNpc()
    local bestObj, bestDist = nil, MAX_DIST + 1
    local playerPos = self_.object.position
    for _, act in ipairs(nearby.actors or {}) do
        if act ~= self_.object and act.type == types.NPC then
            local dead = false
            pcall(function() dead = types.Actor.isDead(act) end)
            if not dead then
                local d = (act.position - playerPos):length()
                if d < bestDist then bestDist = d; bestObj = act end
            end
        end
    end
    return bestObj
end

-- ── UI ────────────────────────────────────────────────────────────────────────

local function headerString()
    if SC.director then return '[СЦЕНА] Что здесь должно произойти?' end
    local n = lockedCtx and (lockedCtx.npc_name ~= '' and lockedCtx.npc_name or lockedCtx.npc_id) or 'NPC'
    return '[ИИ] ' .. n
end

-- The window shows the last few turns of THIS conversation, each on its own
-- line with its speaker. Before, every new line overwrote the previous one —
-- so a bystander cutting in erased the answer you were still reading, and the
-- two ran together into one wall of text.
local sceneLines = {}   -- { {who=..., text=..., emo=...}, ... } newest last

local function pushScene(who, text, emo)
    text = tostring(text or '')
    if text == '' then return end
    sceneLines[#sceneLines + 1] = { who = who, text = text, emo = emo or '', partial = false }
    while #sceneLines > 4 do table.remove(sceneLines, 1) end
end

-- Реплика, которая ещё набирается. Заменяет предыдущую недописанную строку
-- того же говорящего, а не добавляет новую: иначе на экране рос бы столбик из
-- десятка версий одной фразы.
local function pushPartial(who, text)
    text = tostring(text or '')
    if text == '' then return end
    local last = sceneLines[#sceneLines]
    if last and last.partial and last.who == who then
        last.text = text
    else
        sceneLines[#sceneLines + 1] = { who = who, text = text, emo = '', partial = true }
        while #sceneLines > 4 do table.remove(sceneLines, 1) end
    end
end

local function replyString()
    if #sceneLines == 0 then return '(...)' end
    local out = {}
    for _, l in ipairs(sceneLines) do
        local s = l.who .. ': ' .. l.text
        if l.emo ~= '' and l.emo ~= 'neutral' then s = s .. '  [' .. l.emo .. ']' end
        out[#out + 1] = s
    end
    return table.concat(out, '\n\n')
end

local function buildWindow()
    -- autoSize=false обязателен: с включённым автоподбором движок переносит
    -- строки по СВОЕЙ ширине, а не по заданной, и реплика сворачивалась в
    -- колонку в десяток символов, уезжая вниз за границу окна.
    replyWidget = {
        type = ui.TYPE.Text,
        props = { text = replyString(), textSize = 17, textColor = replyColor(),
                  multiline = true, wordWrap = true, autoSize = false,
                  size = v2(SC.REPLY_W, SC.REPLY_H) },
    }
    inputWidget = {
        template = I.MWUI.templates.textEditLine,
        type = ui.TYPE.TextEdit,
        props = { text = inputBuffer, size = v2(SC.REPLY_W, 0) },
        events = {
            textChanged = async:callback(function(text) inputBuffer = text end),
        },
    }
    local inputBox = {
        template = I.MWUI.templates.boxSolid,
        content = ui.content { inputWidget },
    }
    return {
        layer = 'Windows',
        template = I.MWUI.templates.boxSolid,
        props = { anchor = v2(0.5, 0), relativePosition = v2(0.5, 0.03) },
        content = ui.content {{
            template = I.MWUI.templates.padding,
            content = ui.content {{
                type = ui.TYPE.Flex,
                props = { horizontal = false },
                content = ui.content {
                    { type = ui.TYPE.Text,
                      props = { text = headerString(), textSize = 20,
                                textColor = util.color.rgb(1, 0.85, 0.4) } },
                    { external = { grow = 0 }, props = { size = v2(0, 6) }, type = ui.TYPE.Widget },
                    replyWidget,
                    { external = { grow = 0 }, props = { size = v2(0, 8) }, type = ui.TYPE.Widget },
                    inputBox,
                    { external = { grow = 0 }, props = { size = v2(0, 4) }, type = ui.TYPE.Widget },
                    { type = ui.TYPE.Text,
                      props = { text = 'Кликни поле и печатай (Enter — отправить). Esc — вернуть управление. H — закрыть. СКМ — курсор снова.',
                                textSize = 13, textColor = util.color.rgb(0.6, 0.6, 0.6) } },
                    { external = { grow = 0 }, props = { size = v2(0, 4) }, type = ui.TYPE.Widget },
                    { type = ui.TYPE.Text,
                      props = { text = '[ Закрыть ]', textSize = 16,
                                textColor = util.color.rgb(1.0, 0.6, 0.6) },
                      events = { mouseClick = async:callback(function() closeWindow() end) } },
                },
            }},
        }},
    }
end

local function refreshWindow()
    if not isOpen or not window then return end
    if replyWidget then replyWidget.props.text = replyString(); replyWidget.props.textColor = replyColor() end
    if inputWidget then inputWidget.props.text = inputBuffer end
    pcall(function() window:update() end)
end

-- Enter/leave the "live cursor" UI state: Interface mode WITHOUT game pause
-- (via I.UI.setPauseOnMode) and with only the compact Stats panel visible.
-- NOTE: with a fully empty windows list the engine drops the mode and no
-- cursor appears (verified in testing), so one vanilla window must stay.
local function enterCursorMode()
    local mode = (I.UI and I.UI.MODE and I.UI.MODE.Interface) or 'Interface'
    pcall(function() I.UI.setPauseOnMode(mode, false) end)   -- world keeps running
    local ok = pcall(function() I.UI.setMode(mode, { windows = { 'Stats' } }) end)
    if not ok then pcall(function() I.UI.setMode(mode) end) end
end

local function leaveCursorMode()
    local mode = (I.UI and I.UI.MODE and I.UI.MODE.Interface) or 'Interface'
    pcall(function() I.UI.setMode() end)
    -- restore vanilla behaviour: normal Interface (inventory etc.) pauses again
    pcall(function() I.UI.setPauseOnMode(mode, true) end)
end

local cursorOnly   = false  -- middle-mouse free cursor with NO chat window
local cursorActive = false  -- middle-mouse cursor while the chat window is open

-- The chat window is a plain HUD overlay: it exists independently of UI modes.
-- H opens it WITH the typing cursor ready (one click into the field, then just
-- chat: focus persists between messages). Esc returns control (window stays);
-- H closes it once the cursor is off. СКМ re-opens the cursor at any time.
local function openWindow(withCursor)
    isOpen = true
    cursorActive = withCursor and true or false
    if cursorActive then enterCursorMode() end
    window = ui.create(buildWindow())
end

closeWindow = function()
    isOpen = false
    narratorMode = false
    SC.director = false     -- окно закрыли, не дав указания
    if window then pcall(function() window:destroy() end); window = nil end
    replyWidget, inputWidget = nil, nil
    if cursorActive then cursorActive = false; leaveCursorMode() end
    -- Silence any voice line still playing/generating — player walked away.
    reqCounter = reqCounter + 1
    print('[MWAI_REQ] {"type":"stop_voice","req_id":"stop-' .. SESSION_SALT .. '-' .. tostring(reqCounter) .. '"}')
end

-- ── Send dialogue to IPC ──────────────────────────────────────────────────────

-- Emit a tagged request line to openmw.log; the Python bridge tails it.
-- print() works even while the game is paused, unlike global-event delivery.
local function sendRequest(data)
    reqCounter = reqCounter + 1
    data.req_id = 'req-' .. SESSION_SALT .. '-' ..
        tostring(math.floor(core.getSimulationTime())) .. '-' .. reqCounter
    local ok, enc = pcall(json.encode, data)
    if ok then
        print('[MWAI_REQ] ' .. enc)
        -- Start the "is anyone listening?" clock for requests that owe a reply.
        local t = tostring(data.type or 'dialogue')
        if t == 'dialogue' or t == 'narrate' or t == 'voice_stop' then
            waitTimer = 0
        end
    end
end

-- ── Player scene context (time, health, gear, stance) ────────────────────────

-- Diseases/afflictions of an actor, human-readable (для промпта).
local function actorAfflictions(obj)
    local parts = {}
    pcall(function()
        for _, spell in pairs(types.Actor.activeSpells(obj)) do
            local id = tostring(spell.id or ''):lower()
            local rec = core.magic.spells.records[spell.id]
            local st = rec and rec.type or nil
            if id:find('vampire') then
                parts[#parts + 1] = 'ВАМПИР (бледная кожа, глаза хищника — люди в ужасе, стража убивает на месте)'
            elseif id:find('werewolf') or id:find('wolfbane') then
                parts[#parts + 1] = 'ЛИКАНТРОПИЯ (проклятие оборотня — если узнают, затравят как зверя)'
            elseif id:find('corprus') then
                parts[#parts + 1] = 'КОРПРУС (божественная болезнь, люди шарахаются)'
            elseif st == core.magic.SPELL_TYPE.Disease then
                parts[#parts + 1] = 'болен обычной болезнью'
            elseif st == core.magic.SPELL_TYPE.Blight then
                parts[#parts + 1] = 'заражён МОРОВОЙ болезнью'
            end
        end
    end)
    return table.concat(parts, ', ')
end

local function itemName(obj)
    local n = ''
    pcall(function()
        local rec = obj.type and obj.type.record and obj.type.record(obj)
        if rec and rec.name then n = tostring(rec.name) end
    end)
    return n
end

local function buildPlayerContext()
    local parts = {}
    pcall(function()
        local t = core.getGameTime()
        local hour = math.floor((t / 3600) % 24)
        local minute = math.floor((t / 60) % 60)
        local phase
        if hour >= 5 and hour < 11 then phase = 'утро'
        elseif hour >= 11 and hour < 18 then phase = 'день'
        elseif hour >= 18 and hour < 23 then phase = 'вечер'
        else phase = 'глубокая ночь' end
        parts[#parts + 1] = string.format('время %02d:%02d (%s)', hour, minute, phase)
    end)
    pcall(function()
        local rec = types.NPC.record(self_.object)
        if rec then
            local sex = 'мужчина'
            if rec.isMale ~= nil and not rec.isMale then sex = 'ЖЕНЩИНА' end
            parts[#parts + 1] = 'игрок: ' .. sex .. ', раса ' .. tostring(rec.race or '?')
        end
    end)
    pcall(function()
        local lvl = types.Actor.stats.level(self_.object).current
        parts[#parts + 1] = 'уровень ' .. tostring(lvl)
    end)
    pcall(function()
        local h = types.Actor.stats.dynamic.health(self_.object)
        local pct = math.floor(100 * h.current / math.max(1, h.base))
        if pct <= 35 then parts[#parts + 1] = 'игрок тяжело ранен (' .. pct .. '% здоровья)'
        elseif pct <= 75 then parts[#parts + 1] = 'игрок ранен (' .. pct .. '% здоровья)' end
    end)
    pcall(function()
        local SLOT = types.Actor.EQUIPMENT_SLOT
        local eq = types.Actor.getEquipment(self_.object) or {}
        local weapon = eq[SLOT.CarriedRight]
        local wn = weapon and itemName(weapon) or ''
        if wn ~= '' then
            local drawn = false
            pcall(function() drawn = types.Actor.getStance(self_.object) == types.Actor.STANCE.Weapon end)
            parts[#parts + 1] = 'оружие: ' .. wn .. (drawn and ' (ОБНАЖЕНО — держит в руках!)' or ' (в ножнах)')
        else
            parts[#parts + 1] = 'без оружия'
        end
        -- Full visible outfit: an NPC sees a naked outlander or a walking
        -- daedric fortress very differently.
        local worn = {}
        for _, sl in ipairs({ SLOT.Helmet, SLOT.Cuirass, SLOT.Greaves, SLOT.Boots,
                              SLOT.LeftGauntlet, SLOT.Robe, SLOT.Shirt, SLOT.Pants,
                              SLOT.Skirt, SLOT.CarriedLeft, SLOT.Amulet }) do
            local it = sl and eq[sl]
            local nm = it and itemName(it) or ''
            if nm ~= '' then worn[#worn + 1] = nm end
        end
        if #worn == 0 then
            parts[#parts + 1] = 'ИГРОК ПРАКТИЧЕСКИ ГОЛЫЙ (ни брони, ни одежды) — зрелище неприличное и жалкое'
        else
            parts[#parts + 1] = 'одет: ' .. table.concat(worn, ', ')
        end
    end)
    -- Размер штрафа — служебное знание закона, а не общедоступный факт.
    -- Крестьянин в глуши не может знать, что чужак должен Вивеку пять тысяч:
    -- он видит лишь то, что человек держится настороже, и слышал молву.
    pcall(function()
        local bounty = types.Player.getCrimeLevel(self_.object)
        if not bounty or bounty <= 0 then return end
        local lawman = false
        pcall(function()
            local cls = lockedNpcObj and tostring(types.NPC.record(lockedNpcObj).class or ''):lower() or ''
            lawman = cls:find('guard') ~= nil or cls:find('ordinator') ~= nil
        end)
        if lawman then
            parts[#parts + 1] = 'за голову игрока назначен штраф ' ..
                tostring(bounty) .. ' зол. (ты служишь закону и знаешь точную цену)'
        elseif bounty >= 500 then
            parts[#parts + 1] = 'о чужаке ходит дурная слава — говорят, за ним ищет закон ' ..
                '(точной суммы ты не знаешь)'
        end
    end)
    pcall(function()
        local eff = types.Actor.activeEffects(self_.object):getEffect(core.magic.EFFECT_TYPE.Levitate)
        if eff and (eff.magnitude or 0) > 0 then
            parts[#parts + 1] = 'игрок ЛЕВИТИРУЕТ — парит в воздухе на глазах у всех (гость так не входит)'
        end
    end)
    -- Sneaking / magically hidden: an NPC being addressed by someone crouching
    -- in the shadows or half-invisible reacts to THAT, not to the words.
    pcall(function()
        if self_.controls and self_.controls.sneak then
            parts[#parts + 1] = 'игрок КРАДЁТСЯ (пригнулся, прячется в тенях) — и при этом заговаривает'
        end
    end)
    pcall(function()
        local ae = types.Actor.activeEffects(self_.object)
        local inv = ae:getEffect(core.magic.EFFECT_TYPE.Invisibility)
        local cha = ae:getEffect(core.magic.EFFECT_TYPE.Chameleon)
        if inv and (inv.magnitude or 0) > 0 then
            parts[#parts + 1] = 'игрок НЕВИДИМ — голос звучит из пустоты (жутко и подозрительно)'
        elseif cha and (cha.magnitude or 0) > 20 then
            parts[#parts + 1] = 'игрок полупрозрачен от Хамелеона — очертания расплываются'
        end
    end)
    local paff = actorAfflictions(self_.object)
    if paff ~= '' then parts[#parts + 1] = 'состояние игрока: ' .. paff end
    local fight = combatNote()
    if fight ~= '' then parts[#parts + 1] = fight end
    -- Whose walls are these, and is this a decent hour to be inside them?
    pcall(function()
        local owner = interiorOwner()
        if not owner then return end
        local hour = math.floor((core.getGameTime() / 3600) % 24)
        local night = (hour >= 22 or hour < 6)
        local mine = lockedCtx and tostring(lockedCtx.npc_record or ''):lower() == owner
        if mine then
            parts[#parts + 1] = night
                and 'ИГРОК СТОИТ В ТВОЁМ ДОМЕ ПОСРЕДИ НОЧИ — как он сюда попал и что ему нужно?!'
                or  'разговор происходит В ТВОЁМ ДОМЕ (это твои стены и твоё добро вокруг)'
        else
            parts[#parts + 1] = night
                and 'вы оба в ЧУЖОМ жилище глубокой ночью — хозяин был бы не рад'
                or  'вы в чужом частном жилище'
        end
    end)
    pcall(function()
        if companionLoss and core.getSimulationTime() < (companionLoss.until_t or 0) then
            parts[#parts + 1] = 'НЕДАВНО НА ГЛАЗАХ ИГРОКА ПОГИБ ЕГО СПУТНИК ' ..
                tostring(companionLoss.name) .. ' — эта смерть реальна и свежа'
        end
    end)
    pcall(function()
        local cell = self_.object.cell
        if cell then
            if cell.region then
                local rr = core.regions.records[cell.region]
                parts[#parts + 1] = 'регион: ' .. tostring(rr and rr.name or cell.region)
            end
            local w = core.weather.getCurrent(cell)
            if w and w.name then
                local wn2 = tostring(w.name)
                parts[#parts + 1] = 'погода: ' .. wn2 .. ((w.isStorm and ' (БУРЯ!)') or '')
            end
        end
    end)
    return table.concat(parts, '; ')
end

-- Health/afflictions of the locked NPC — grounds wounded-enemy talk (mercy!).
local function npcConditionString(obj)
    local parts = {}
    pcall(function()
        local h = types.Actor.stats.dynamic.health(obj)
        local pct = math.floor(100 * h.current / math.max(1, h.base))
        if pct <= 25 then parts[#parts + 1] = 'ТЯЖЕЛО РАНЕН (' .. pct .. '% здоровья) — на грани смерти'
        elseif pct <= 60 then parts[#parts + 1] = 'ранен (' .. pct .. '% здоровья)' end
    end)
    local aff = actorAfflictions(obj)
    if aff ~= '' then parts[#parts + 1] = aff end
    return table.concat(parts, '; ')
end

-- Who is within earshot: NPC speak differently in front of guards/masters.
local function buildBystanders()
    local names = {}
    pcall(function()
        local ppos = self_.object.position
        for _, act in ipairs(nearby.actors or {}) do
            if act ~= self_.object and act ~= lockedNpcObj
               and act.type == types.NPC and #names < 8 then
                local d = (act.position - ppos):length()
                if d <= 700 then
                    local rec = nil
                    pcall(function() rec = types.NPC.record(act) end)
                    local n = rec and tostring(rec.name or '') or ''
                    if n ~= '' then
                        local cls = rec and tostring(rec.class or '') or ''
                        local tag = ''
                        if cls:lower():find('guard') then tag = ' (СТРАЖНИК!)' end
                        if act == companionObj then tag = ' (спутник игрока)' end
                        names[#names + 1] = n .. tag
                    end
                end
            end
        end
    end)
    return table.concat(names, ', ')
end

-- Чем NPC рискует, если согласится на тёмное дело. Раньше согласие зависело
-- только от морали модели, и трактирщик травил человека за двести золотых, не
-- задумываясь ни о страже за спиной, ни о том, что его узнают. Считаем по
-- РЕАЛЬНОЙ обстановке, а решение всё равно остаётся за характером.
local function riskNote()
    local guards, folk, nearestGuard = 0, 0, 1e9
    pcall(function()
        local ppos = self_.object.position
        for _, act in ipairs(nearby.actors or {}) do
            if act ~= self_.object and act ~= lockedNpcObj and act.type == types.NPC
               and not types.Actor.isDead(act) then
                local d = (act.position - ppos):length()
                if d <= 1600 then
                    local cls = ''
                    pcall(function() cls = tostring(types.NPC.record(act).class or ''):lower() end)
                    if cls:find('guard') or cls:find('ordinator') then
                        guards = guards + 1
                        if d < nearestGuard then nearestGuard = d end
                    elseif d <= 700 then
                        folk = folk + 1
                    end
                end
            end
        end
    end)

    local bits = {}
    if guards > 0 then
        bits[#bits + 1] = 'стража рядом (' .. guards .. ', ближайший в ' ..
            math.floor(nearestGuard / 64) .. ' шагах)'
    end
    if folk > 0 then
        bits[#bits + 1] = 'посторонних глаз поблизости: ' .. folk
    end
    if guards == 0 and folk == 0 then
        bits[#bits + 1] = 'вокруг ни души — свидетелей не будет'
    end

    -- Известность игрока как преступника: с таким и говорить опасно.
    local bounty = 0
    pcall(function() bounty = types.Player.getCrimeLevel(self_.object) or 0 end)
    if bounty >= 1000 then
        bits[#bits + 1] = 'за этим чужаком уже назначена немалая цена (' .. bounty .. ')'
    elseif bounty > 0 then
        bits[#bits + 1] = 'у чужака есть долг перед законом (' .. bounty .. ')'
    end

    -- Своё жильё против чужого: дома и стены помогают.
    pcall(function()
        local cell = self_.object.cell
        if cell and not cell.isExterior then
            local owner = interiorOwner()
            local mine = ''
            pcall(function()
                mine = lockedNpcObj and tostring(lockedNpcObj.recordId or ''):lower() or ''
            end)
            if owner and owner == mine then
                bits[#bits + 1] = 'вы у тебя дома, здесь ты хозяин положения'
            elseif owner then
                bits[#bits + 1] = 'вы в чужом жилище'
            end
        end
    end)

    return 'ЧЕМ ТЫ РИСКУЕШЬ: ' .. table.concat(bits, '; ') .. '.'
end

-- Active (started, not finished) quest ids + stages — the ids are canonical
-- Morrowind quest ids (e.g. "ms_fargothring"), the LLM knows the lore behind them.
-- Дела игрока делятся надвое: те, где замешан САМ этот NPC (его имя стоит в
-- записи журнала — он о них знает и может подсказать по делу), и все прочие,
-- о которых он знать не может. Раньше в голову каждому встречному вываливался
-- весь журнал целиком, и трактирщик рассуждал о поручениях Клинков.
local function buildQuestList()
    local mine, others = {}, 0
    local myName = ''
    pcall(function()
        myName = lockedNpcObj and tostring(types.NPC.record(lockedNpcObj).name or ''):lower() or ''
    end)
    pcall(function()
        for qid, q in pairs(types.Player.quests(self_.object)) do
            local started, finished, stage = false, false, 0
            pcall(function() started = q.started; finished = q.finished; stage = q.stage end)
            if started and not finished then
                local involved = false
                if myName ~= '' and #myName > 3 then
                    pcall(function()
                        local rec = core.dialogue.journal.records[qid]
                        for _, info in ipairs(rec and rec.infos or {}) do
                            if info.questStage == stage and info.text
                               and tostring(info.text):lower():find(myName, 1, true) then
                                involved = true
                                break
                            end
                        end
                    end)
                end
                if involved and #mine < 6 then
                    mine[#mine + 1] = tostring(qid) .. ':' .. tostring(stage)
                else
                    others = others + 1
                end
            end
        end
    end)
    local s = table.concat(mine, ', ')
    if others > 0 then
        s = s .. (s ~= '' and ' | ' or '') ..
            'кроме этого чужак занят ещё чем-то (' .. others ..
            ' дел), но тебе о них знать неоткуда'
    end
    return s
end

-- ── Canonical ESM dialogue grounding ─────────────────────────────────────────
-- One-time index: NPC recordId -> up to 8 of their PERSONAL vanilla lines
-- (topic/greeting infos with filterActorId == this NPC). Fed to the LLM as
-- the source of truth so it stops inventing quests this NPC never had.

local canonIndex = nil

local function buildCanonIndex()
    canonIndex = {}
    local t0 = core.getRealTime and core.getRealTime() or 0
    local count = 0
    pcall(function()
        for _, group in ipairs({ core.dialogue.topic, core.dialogue.greeting }) do
            for _, recdlg in ipairs(group.records) do
                local infos = recdlg.infos
                if infos then
                    for _, info in ipairs(infos) do
                        local aid = info.filterActorId
                        if aid and aid ~= '' and info.text and info.text ~= '' then
                            aid = tostring(aid):lower()
                            local bucket = canonIndex[aid]
                            if not bucket then bucket = {}; canonIndex[aid] = bucket end
                            if #bucket < 8 then
                                bucket[#bucket + 1] =
                                    '[' .. tostring(recdlg.name or '?') .. '] ' ..
                                    string.sub(tostring(info.text), 1, 220)
                                count = count + 1
                            end
                        end
                    end
                end
            end
        end
    end)
    print('[morrowind-ai] canon index built: ' .. tostring(count) .. ' personal lines')
end

local function canonFor(npcId)
    if canonIndex == nil then buildCanonIndex() end
    local bucket = canonIndex[tostring(npcId or ''):lower()]
    if not bucket or #bucket == 0 then return '' end
    return table.concat(bucket, '\n')
end

-- Is this person load-bearing for the story? Canon dialogue alone was too
-- narrow a test: plenty of quest NPCs have no lines of their own but carry an
-- attached script, or are the target of a quest the player has open right now.
-- Move or kill one of those and a quest silently dies forever.
local function isStoryCritical(obj, ctx)
    if ctx and canonFor(ctx.npc_record) ~= '' then return true, 'канонные реплики' end
    local guarded, why = false, ''
    pcall(function()
        local rec = types.NPC.record(obj)
        -- Скрипт на персонаже почти всегда означает участие в квесте.
        if rec and tostring(rec.mwscript or '') ~= '' then
            guarded, why = true, 'на нём висит скрипт'
            return
        end
        -- Имя упомянуто в тексте активной записи журнала.
        local nm = tostring(rec and rec.name or ''):lower()
        if nm == '' or #nm < 4 then return end
        for _, q in ipairs(types.Player.quests(self_.object)) do
            if not q.finished then
                local qr = core.dialogue.journal.records[q.id]
                for _, info in ipairs(qr and qr.infos or {}) do
                    if info.questStage == q.stage and info.text
                       and tostring(info.text):lower():find(nm, 1, true) then
                        guarded, why = true, 'он нужен в текущем задании'
                        return
                    end
                end
            end
        end
    end)
    return guarded, why
end

-- Companion info for the request — only when they are alive, valid and nearby.
local function companionFields()
    if not companionObj or not companionCtx then return nil end
    local ok = false
    pcall(function()
        ok = companionObj:isValid()
            and not types.Actor.isDead(companionObj)   -- the dead follow no one
            and (companionObj.position - self_.object.position):length() < 1800
    end)
    if not ok then return nil end
    return companionCtx
end

-- Narrator channel: questions to the voice-over, grounded in the live scene.
local function sendNarrate(text)
    sendRequest({
        type = 'narrate',
        player_text = tostring(text or ''),
        location = (self_.object.cell and tostring(self_.object.cell.name or '')) or '',
        player_context = buildPlayerContext(),
        bystanders = buildBystanders(),
        conversation_history = recentHistory('narrator'),
    })
    if text ~= '' then pushHistory('narrator', 'player', text) end
end

-- Eavesdroppers: who is close enough to actually HEAR this line.
-- lastListeners get a memory of the exchange; a "mentioned" listener (their
-- name or their office — e.g. страж* for guards — comes up) may intervene.
local lastListeners   = {}     -- { {id=..., name=...}, ... } for memory notes
local lastListenerObj = nil    -- the mentioned one, for real intervention
local lastPlayerLine  = ''

local function scanListeners(text)
    lastListeners, lastListenerObj = {}, nil
    local mentioned = nil
    local low = tostring(text or ''):lower()
    pcall(function()
        local ppos = self_.object.position
        for _, act in ipairs(nearby.actors or {}) do
            if act ~= self_.object and act ~= lockedNpcObj and act ~= companionObj
               and act.type == types.NPC and #lastListeners < 3 then
                local d = (act.position - ppos):length()
                if d <= 450 and not types.Actor.isDead(act) then
                    local rec = types.NPC.record(act)
                    local nm = rec and tostring(rec.name or '') or ''
                    if nm ~= '' then
                        lastListeners[#lastListeners + 1] = { id = tostring(act.id), name = nm }
                        if not mentioned then
                            local first = nm:match('^[^%s]+') or nm
                            local isGuard = tostring(rec.class or ''):lower():find('guard') ~= nil
                            if (first:len() > 3 and low:find(first:lower(), 1, true))
                               or (isGuard and low:find('страж', 1, true)) then
                                mentioned = act
                            end
                        end
                    end
                end
            end
        end
    end)
    lastListenerObj = mentioned
    return mentioned
end

-- What an NPC carries right now, as a prompt string (rebuilt every message so
-- it can never promise an item that already changed hands).
local function collectInventory(npc)
    local out = ''
    pcall(function()
        local names, seen = {}, {}
        for _, it in ipairs(types.Actor.inventory(npc):getAll() or {}) do
            if #names >= 14 then break end
            local nm = ''
            pcall(function()
                local rec = it.type and it.type.record and it.type.record(it)
                nm = rec and tostring(rec.name or '') or ''
            end)
            if nm ~= '' and not seen[nm] then seen[nm] = true; names[#names + 1] = nm end
        end
        out = table.concat(names, '; ')
    end)
    return out
end

local function buildDialoguePayload(text, rtype, mentionedObj)
    -- Engine disposition right now (0-100) — read fresh each message.
    local disp = nil
    pcall(function()
        if lockedNpcObj and lockedNpcObj:isValid() then
            disp = types.NPC.getDisposition(lockedNpcObj, self_.object)
        end
    end)
    local payload = {
        type = rtype or 'dialogue',
        npc_id = lockedCtx.npc_id, npc_name = lockedCtx.npc_name, npc_race = lockedCtx.npc_race,
        npc_class = lockedCtx.npc_class, npc_faction = lockedCtx.npc_faction,
        npc_is_male = lockedCtx.npc_is_male,
        npc_disposition = disp,
        -- How far the speaker is, so their voice can be as loud as real life.
        distance = (function()
            local d = 0
            pcall(function()
                if lockedNpcObj then d = (lockedNpcObj.position - self_.object.position):length() end
            end)
            return math.floor(d)
        end)(),
        location = lockedCtx.location, player_text = text,
        -- Authoritative, save-scoped memory (overrides the bridge's own store):
        conversation_history = recentHistory(lockedCtx.npc_id),
        player_context = buildPlayerContext(),
        active_quests  = buildQuestList(),
        bystanders     = buildBystanders(),
        -- Bodies in view and whose roof they are under: every other helper
        -- skips the dead and tells nobody where they are standing, so an NPC
        -- could kill a man in a stranger's house and then claim to be at home
        -- with no idea of any corpse.
        corpses        = corpsesNote(),
        npc_place      = (lockedNpcObj and npcPlaceNote(lockedNpcObj)) or '',
        -- Чем NPC рискует прямо сейчас: стража, чужие глаза, репутация игрока.
        risk_note      = riskNote(),
        -- Что стало с человеком после того, как разговор изменил его жизнь.
        npc_fate       = fateNote(lockedCtx.npc_id),
        -- Спутник: своя скрытая история, открытая ровно настолько, насколько
        -- далеко зашёл сам игрок.
        is_companion   = (companionObj ~= nil and lockedNpcObj == companionObj),
        companion_arc  = npcArc[lockedCtx.npc_id],
        arc_reveal     = storyProgress(),
        npc_condition  = (lockedNpcObj and npcConditionString(lockedNpcObj)) or '',
        -- Save-scoped social state (bridge keeps NO cross-save memory of these):
        npc_last_mood  = npcMood[lockedCtx.npc_id],
        npc_life_facts = npcFacts[lockedCtx.npc_id],
        npc_canon      = canonFor(lockedCtx.npc_record),
        -- Refreshed EVERY line: a one-shot snapshot went stale as soon as
        -- anything changed hands, and the model offered things already gone.
        npc_inventory  = (lockedNpcObj and collectInventory(lockedNpcObj)) or '',
        deal_note      = (function()
            local id = lockedCtx.npc_id
            if escort and escort.npc_id == id then
                local day = math.floor((core.getGameTime() or 0) / 86400)
                return 'Ты нанял игрока довести тебя до ' .. escort.townRu .. ' за ' ..
                       escort.reward .. ' зол.; осталось дней: ' .. ((escort.dueDay or day) - day) ..
                       '. Вы в пути.'
            end
            if duel and duel.npc_id == id then
                return 'С игроком идёт ДУЭЛЬ ЧЕСТИ, заклад ' .. duel.stake .. ' зол.'
            end
            return ''
        end)(),
        debt_note      = (function()
            local d = debts[lockedCtx.npc_id]
            if not d or (d.amount or 0) <= 0 then return '' end
            local day = math.floor((core.getGameTime() or 0) / 86400)
            local left = (d.due or day) - day
            if left >= 0 then
                return 'Игрок должен тебе ' .. d.amount .. ' зол.; срок возврата через ' .. left .. ' дн.'
            end
            return 'Игрок ПРОСРОЧИЛ долг тебе на ' .. (-left) .. ' дн.: ' .. d.amount ..
                   ' зол. Требуй своё — по-своему: уговором, угрозой или через закон.'
        end)(),
        rumors         = rumorsHere(),
    }
    local comp = companionFields()
    if comp and comp.npc_id ~= lockedCtx.npc_id then
        payload.companion_id      = comp.npc_id
        payload.companion_name    = comp.npc_name
        payload.companion_race    = comp.npc_race
        payload.companion_class   = comp.npc_class
        payload.companion_is_male = comp.npc_is_male
    end
    -- Listener candidate: the NPC that was named, else the closest eavesdropper.
    -- The bridge decides whether they actually butt in — by name-mention or by
    -- the interlocutor's HEARD:alarm verdict on what the player just said.
    local listenerObj = mentionedObj
    if not listenerObj then
        listenerObj = (lastListeners[1] and findActorById(lastListeners[1].id)) or nil
    end
    if listenerObj then
        local lc = buildNpcContext(listenerObj)
        payload.listener_id        = lc.npc_id
        payload.listener_name      = lc.npc_name
        payload.listener_race      = lc.npc_race
        payload.listener_class     = lc.npc_class
        payload.listener_is_male   = lc.npc_is_male
        payload.listener_mentioned = (mentionedObj ~= nil)
    end
    return payload
end

local function sendMessage(text)
    if narratorMode then sendNarrate(text); return end
    if not lockedCtx then showMsg('[ИИ] Никто не выбран; нажми H рядом с NPC.'); return end
    -- Service tokens (__greet__/__theft__:X/__surrender__/...) must never be
    -- stored as the player's words (audit 3.3).
    local isService = text == '' or text:sub(1, 2) == '__'
    lastPlayerLine = (not isService) and text or ''
    local mentionedObj = scanListeners(isService and '' or text)
    sendRequest(buildDialoguePayload(text, 'dialogue', mentionedObj))
    if not isService then
        pushHistory(lockedCtx.npc_id, 'player', text)
    end
end


-- Voice mode: lock the aimed NPC silently and ask the bridge to LISTEN.
-- No windows at all — the whole exchange lives in subtitles + voice.
local function startVoiceExchange(npc)
    local vId = ''
    pcall(function() vId = tostring(npc.id or '') end)
    if not lockedCtx or lockedCtx.npc_id ~= vId then sceneLines = {} end
    -- Собеседника больше НЕ замораживаем. Раньше на него вешался пакет
    -- «стой на месте» на 7200 — а duration у Wander считается в ИГРОВЫХ ЧАСАХ,
    -- то есть триста суток, и снять его было нечем: обработчика на конец
    -- разговора нет. Каждый, с кем игрок хоть раз заговорил, замирал до конца
    -- прохождения. Затевалось это, когда ответ шёл десятки секунд и NPC успевал
    -- уйти на полуслове; сейчас от вопроса до звука меньше трёх секунд.
    lockedCtx    = buildNpcContext(npc)
    lockedNpcObj = npc
    lastPlayerLine = ''
    scanListeners('')          -- eavesdroppers still get memory of the exchange
    voiceTalking = true
    sendRequest({ type = 'voice_start' })
    showMsg('● ГОВОРИ, держи V — ' .. (lockedCtx.npc_name ~= '' and lockedCtx.npc_name or '???'))
end

-- Key released: send what was recorded through the normal dialogue pipeline.
local function finishVoiceExchange()
    if not voiceTalking then return end
    voiceTalking = false
    showMsg('… отправляю')
    sendRequest(buildDialoguePayload('', 'voice_stop', nil))
end

local function lockAndGreet(npc)
    -- A new interlocutor starts a fresh transcript: the previous conversation
    -- must never bleed into this one on screen.
    local newId = ''
    pcall(function() newId = tostring(npc.id or '') end)
    if not lockedCtx or lockedCtx.npc_id ~= newId then sceneLines = {} end
    -- Собеседника НЕ замораживаем — см. подробности в startVoiceExchange.
    -- Коротко: пакет «стой на месте» ставился на 7200 ИГРОВЫХ ЧАСОВ (триста
    -- суток) и не снимался никогда, так что мир постепенно застывал.
    lockedCtx     = buildNpcContext(npc)
    lockedNpcObj  = npc
    lastReplyText = '(здоровается...)'
    lastSpeaker   = ''
    lastEmotion   = ''
    inputBuffer   = ''
    sendRequest({
        type = 'lock_npc',
        npc_id = lockedCtx.npc_id, npc_name = lockedCtx.npc_name, npc_race = lockedCtx.npc_race,
        npc_class = lockedCtx.npc_class, npc_faction = lockedCtx.npc_faction, location = lockedCtx.location,
        npc_is_male = lockedCtx.npc_is_male,
    })
    sendMessage('__greet__')
    if isOpen then refreshWindow() else openWindow(true) end
end

-- ── Input ─────────────────────────────────────────────────────────────────────
-- While open, only physical Enter/Esc are handled here (they are not typeable
-- characters, so they never collide with Russian/Latin text going to the field).
-- H is handled only when closed, so typing a letter on the H key can't re-trigger.

local function onKeyPress(key)
    if not key then return end
    local code = key.code

    if not isOpen then
        if code == HAIL_KEY then
            local npc = findNearestNpc()
            if not npc then showMsg('[ИИ] Рядом нет NPC.'); return end
            lockAndGreet(npc)
        elseif code == VOICE_KEY then
            -- Push-to-talk: hold V, speak, release. Talks to whoever you look at.
            if voiceTalking then return end
            local npc = findAimedNpc() or findNearestNpc()
            if not npc then showMsg('[ИИ] Никого не видно — посмотри на собеседника.'); return end
            startVoiceExchange(npc)
        elseif code == NARRATOR_KEY then
            -- Open a conversation with the Narrator — works anywhere, no NPC.
            narratorMode = true
            lockedNpcObj = nil
            lockedCtx = { npc_id = 'narrator', npc_record = '', npc_name = 'Рассказчик',
                          npc_race = '', npc_class = '', npc_faction = '',
                          location = (self_.object.cell and tostring(self_.object.cell.name or '')) or '',
                          npc_is_male = true }
            lastReplyText = '(рассказчик присматривается к сцене...)'
            lastSpeaker, lastEmotion, inputBuffer = '', '', ''
            sendNarrate('')
            openWindow(true)
        elseif code == input.KEY.K then
            -- Режиссёр: сам задаёшь, что здесь произойдёт. Отдельная клавиша,
            -- чтобы не путалось с обычным разговором по H.
            if SC.cur then showMsg('[сцена] Одна уже идёт.'); return end
            SC.director   = true
            narratorMode  = false
            lockedNpcObj  = nil
            lockedCtx     = nil
            lastReplyText = '(назови участников по именам и скажи, что между ними произойдёт)'
            lastSpeaker, lastEmotion, inputBuffer = '', '', ''
            sceneLines = {}
            openWindow(true)
        end
        return
    end

    -- Window open. Esc is NOT handled here — the engine gets it (vanilla menu).
    if (code == HAIL_KEY or code == NARRATOR_KEY) and not cursorActive then
        -- H closes the chat only when the typing cursor is off, so typing
        -- letters on the H key can never close the window mid-sentence.
        closeWindow()
        return
    end

    if cursorActive and (code == input.KEY.Enter or code == SEND_KEY) then
        if inputBuffer ~= '' then
            local sent = inputBuffer
            -- Указание режиссёра: не реплика собеседнику, а задание на сцену.
            if SC.director then
                inputBuffer = ''
                SC.director = false
                closeWindow()
                if SC.ask('', sent) then
                    showMsg('[сцена] Ставлю: ' .. sent)
                end
                return
            end
            -- Maintenance command: wipe ALL AI memory of this save (history,
            -- moods, life facts, rumors). Type /wipe in the field and Enter.
            if sent == '/wipe' then
                npcHistory, npcMood, npcFacts, worldRumors = {}, {}, {}, {}
                inputBuffer = ''
                sceneLines = {}
                pushScene('[система]', 'память ИИ очищена: истории, настроения, факты, слухи', '')
                refreshWindow()
                showMsg('[ИИ] Память очищена.')
                return
            end
            -- /осмотреться — narrator mode: describe the scene like a text quest.
            if sent == '/осмотреться' or sent == '/look' then
                inputBuffer = ''
                pushScene('Ты', '(оглядываешься по сторонам)', '')
                refreshWindow()
                sendRequest({
                    type = 'narrate',
                    location = (lockedCtx and lockedCtx.location) or
                        (self_.object.cell and tostring(self_.object.cell.name or '')) or '',
                    player_context = buildPlayerContext(),
                    bystanders = buildBystanders(),
                })
                return
            end
            -- /history — technical peek at what this NPC actually remembers.
            if sent == '/history' or sent == '/история' then
                inputBuffer = ''
                local h = npcHistory[lockedCtx and lockedCtx.npc_id or ''] or {}
                if #h == 0 then
                    pushScene('[память]', 'этот персонаж ещё ничего о тебе не помнит', '')
                else
                    local lines = {}
                    for i = math.max(1, #h - 6), #h do
                        local who = (h[i].role == 'player') and 'Ты' or
                            ((lockedCtx and lockedCtx.npc_name ~= '' and lockedCtx.npc_name) or 'NPC')
                        lines[#lines + 1] = who .. ': ' .. string.sub(tostring(h[i].content or ''), 1, 110)
                    end
                    pushScene('[память, ' .. #h .. ' реплик]', table.concat(lines, '\n'), '')
                end
                refreshWindow()
                return
            end
            sendMessage(sent)
            pushScene('Ты', sent, '')   -- your own words stay on screen
            inputBuffer = ''
            refreshWindow()
        end
    end
end

-- Releasing the talk key ends the recording and sends it.
local function onKeyRelease(key)
    if key and key.code == VOICE_KEY and voiceTalking then
        pcall(finishVoiceExchange)
    end
end

-- ── Reply polling (runs during pause via onFrame) ──────────────────────────────

-- ── Real NPC actions (wired to OpenMW AI packages via StartAIPackage) ─────────
-- The builtin scripts/omw/ai.lua on every NPC handles the 'StartAIPackage'
-- event, so we can start Combat/Follow/Wander from here. AI packages tick only
-- when the game is unpaused, so for attack we close the chat window.

local function npcName()
    return (lockedCtx and lockedCtx.npc_name ~= '' and lockedCtx.npc_name) or 'NPC'
end

-- Find a nearby NPC whose display name matches `name` (exact or substring).
local function findNpcByName(name)
    name = tostring(name or '')
    if name == '' or name == 'none' then return nil end
    local found = nil
    for _, act in ipairs(nearby.actors or {}) do
        if act ~= self_.object and act.type == types.NPC and act ~= lockedNpcObj then
            pcall(function()
                local rec = types.NPC.record(act)
                local n = rec and tostring(rec.name or '') or ''
                if n ~= '' and (n == name or n:find(name, 1, true) or name:find(n, 1, true)) then
                    found = act
                end
            end)
            if found then return found end
        end
    end
    return nil
end

-- ── Contracts: escort & duel ─────────────────────────────────────────────────
-- Both are real state machines settled by the engine: gold physically moves,
-- arrival is checked against the actual cell, the duel ends on real health
-- thresholds. Nothing here depends on the LLM keeping its word.

local RU_TOWN_CELL = {
    ['балмор'] = 'balmora', ['вивек'] = 'vivec', ["альд'рун"] = 'ald', ['альд-рун'] = 'ald',
    ['альдрун'] = 'ald', ['садрит'] = 'sadrith', ['гнисис'] = 'gnisis', ['кальдер'] = 'caldera',
    ['пелагиад'] = 'pelagiad', ['сейда'] = 'seyda', ['молаг'] = 'molag', ['суран'] = 'suran',
    ['хла оад'] = 'hla oad', ['гнаар'] = 'gnaar', ['дагон'] = 'dagon fel', ['вос'] = 'vos',
    ['эбенгард'] = 'ebonheart', ['маар'] = 'maar gan', ['хуул'] = 'khuul', ['тель'] = 'tel ',
}

local function townCellKey(ru)
    local low = tostring(ru or ''):lower()
    for k, v in pairs(RU_TOWN_CELL) do
        if low:find(k, 1, true) then return v end
    end
    return nil
end

local function playerCellName()
    local n = ''
    pcall(function()
        local c = self_.object.cell
        if c then n = tostring(c.name or ''):lower() end
    end)
    return n
end

local function healthFrac(obj)
    local f = 1.0
    pcall(function()
        local h = types.Actor.stats.dynamic.health(obj)
        f = h.current / math.max(1, h.base)
    end)
    return f
end

-- Start an escort contract: the NPC follows and pays on arrival.
local function startEscort(obj, name, townRu, reward)
    local key = townCellKey(townRu)
    if not key then
        showMsg('(такого города нет на карте — уговор не состоялся)')
        return false
    end
    local day = math.floor((core.getGameTime() or 0) / 86400)
    escort = {
        npc_id = tostring(obj.id), name = name, cellKey = key, townRu = tostring(townRu),
        reward = math.max(1, math.min(1000, math.floor(tonumber(reward) or 0))),
        dueDay = day + 6,
    }
    pcall(function()
        obj:sendEvent('StartAIPackage', { type = 'Follow', target = self_.object, cancelOther = true })
    end)
    companionObj, companionCtx = obj, lockedCtx
    showMsg('Уговор: довести ' .. name .. ' до ' .. escort.townRu ..
            ' — награда ' .. escort.reward .. ' зол. (6 дней)')
    pushHistory(escort.npc_id, 'npc',
        '(ФАКТ: заключён уговор — игрок ведёт тебя в ' .. escort.townRu ..
        ', по прибытии ты платишь ' .. escort.reward .. ' зол.)')
    return true
end

-- Start a duel: BOTH stakes are physically escrowed into the opponent, the
-- engine runs the fight and pays the winner.
local function startDuel(obj, name, stake)
    stake = math.max(1, math.min(1000, math.floor(tonumber(stake) or 0)))
    local have = 0
    pcall(function() have = types.Actor.inventory(self_.object):countOf('gold_001') or 0 end)
    if have < stake then
        showMsg('(на такую ставку у тебя нет золота — дуэль не состоялась)')
        return false
    end
    core.sendGlobalEvent('MorrowindAiTakeGold', { amount = stake, npc = obj })
    duel = { npc_id = tostring(obj.id), name = name, stake = stake,
             started = core.getSimulationTime() }
    pcall(function()
        obj:sendEvent('StartAIPackage', { type = 'Combat', target = self_.object, cancelOther = true })
    end)
    showMsg('ДУЭЛЬ ЧЕСТИ: ' .. name .. ', ставка ' .. stake ..
            ' зол. Бой до первой серьёзной крови.')
    pushHistory(duel.npc_id, 'npc',
        '(ФАКТ: идёт дуэль чести с игроком, ставка ' .. stake .. ' зол.)')
    closeWindow()
    return true
end

-- Armed ultimatums (ACTION:threaten): npc_id -> {obj, name, until_t, bounty0}.
-- Enforced by threatWatch(): draw a weapon near them or raise your bounty
-- while armed — they attack for real. Session-scoped.
local armedThreats = {}

-- actorObj: who performs the action (locked NPC by default, or the companion).
-- hostileTarget: whom 'attack' is aimed at (the player for the locked NPC;
-- the locked NPC when the companion intervenes on the player's side).
-- ── Расследование тёмных дел ────────────────────────────────────────────────
-- Нанять исполнителя было безопаснее, чем убивать самому: свидетелей у сговора
-- не было вовсе, и мир никогда не связывал игрока с преступлением. Теперь у
-- каждого дела есть очевидцы и срок, через который оно всплывает.
-- Живёт в сейве: загрузка отменяет и расследование.
local dirtyDeeds = {}

local DEED_WORDS = {
    poison = 'отравление',  steal = 'кража',   plant = 'подброшенная улика',
    frame  = 'подстава',    abduct = 'похищение человека',
}

local function recordDeed(kind, perpObj, perpName, victimName)
    local seen = {}
    pcall(function()
        for _, w in ipairs(nearby.actors or {}) do
            if w ~= perpObj and w ~= self_.object and w.type == types.NPC
               and not types.Actor.isDead(w) and #seen < 4 then
                local d = (w.position - perpObj.position):length()
                -- Видит и исполнителя, и игрока рядом с ним — значит свяжет одно с другим.
                if d < 800 and hasLineOfSight(w) then
                    seen[#seen + 1] = tostring(w.id)
                end
            end
        end
    end)
    dirtyDeeds[#dirtyDeeds + 1] = {
        kind = kind, victim = tostring(victimName or ''),
        perpName = tostring(perpName or ''),
        at = core.getGameTime() or 0,
        cell = (self_.object.cell and tostring(self_.object.cell.name or '')) or '',
        seen = seen, exposed = false,
    }
    if #seen > 0 then
        showMsg('(кто-то рядом это видел)')
    end
end

-- Проходит время — и дело всплывает, если были очевидцы. Без свидетелей оно
-- так и останется тайной: это и есть награда за то, что игрок всё продумал.
local DEED_HOURS = 8

local function investigateDeeds()
    local now = core.getGameTime() or 0
    for _, d in ipairs(dirtyDeeds) do
        if (not d.exposed) and #d.seen > 0 and (now - d.at) > DEED_HOURS * 3600 then
            d.exposed = true
            local what = DEED_WORDS[d.kind] or 'тёмное дело'
            addRumor('в ' .. (d.cell ~= '' and d.cell or 'округе') .. ' раскрылось ' ..
                what .. (d.victim ~= '' and (' — пострадал(а) ' .. d.victim) or '') ..
                '; чужака видели с ' .. (d.perpName ~= '' and d.perpName or 'исполнителем') ..
                ' незадолго до случившегося')
            -- Настоящее последствие: закон вешает на игрока штраф.
            core.sendGlobalEvent('MorrowindAiReportCrime', {})
            showMsg('Твоё дело всплыло: ' .. what .. '. Свидетели заговорили.')
            for _, wid in ipairs(d.seen) do
                pushHistory(wid, 'npc', '(ФАКТ: ты видел(а), как чужак сговаривался с ' ..
                    (d.perpName ~= '' and d.perpName or 'кем-то') ..
                    ', и вскоре случилось ' .. what .. '. Ты связал(а) одно с другим.)')
            end
        end
    end
    while #dirtyDeeds > 20 do table.remove(dirtyDeeds, 1) end
end

-- Барьер между «модель сказала» и «движок сделал». Без него уговорённый NPC
-- мог отравить квестового персонажа, увести торговца или натравить стражу на
-- того, кто нужен в задании — и сейв тихо становился непроходимым.
-- Возвращает (запрещено?, что показать игроку).
local DIRTY_COOLDOWN = 45          -- секунд между тёмными делами одного NPC
local lastDirtyAt = 0

local function victimProtected(victim, kind)
    if not victim then return false, '' end
    local okValid = false
    pcall(function() okValid = victim:isValid() and not types.Actor.isDead(victim) end)
    if not okValid then return true, 'Поздно: с этим уже ничего не сделать.' end

    -- Спутника не трогаем: игрок сам его ведёт.
    if companionObj and victim == companionObj then
        return true, 'Твой спутник — не мишень для такого.'
    end
    local nm = ''
    pcall(function() nm = tostring(types.NPC.record(victim).name or '') end)
    local guarded = isStoryCritical(victim, { npc_record = (function()
        local r = ''
        pcall(function() r = tostring(victim.recordId or '') end)
        return r
    end)() })
    if guarded then
        return true, (nm ~= '' and nm or 'Этот человек') ..
            ' слишком заметная фигура — такое сойдёт с рук лишь на словах.'
    end
    -- Стражу нельзя подставлять: она же и расследует.
    if kind == 'frame' then
        local isGuard = false
        pcall(function()
            isGuard = tostring(types.NPC.record(victim).class or ''):lower():find('guard') ~= nil
        end)
        if isGuard then return true, 'Подставить стражника — то же, что донести на себя.' end
    end
    -- Кулдаун: череда злодейств подряд превращает мир в меню услуг.
    local now = core.getSimulationTime()
    if now - lastDirtyAt < DIRTY_COOLDOWN then
        return true, 'Слишком скоро после прошлого дела — надо переждать.'
    end
    lastDirtyAt = now
    return false, ''
end

local function execAction(action, target, actorObj, actorName, hostileTarget, cond, fate)
    if action == 'none' or action == '' then return end
    local obj = actorObj or lockedNpcObj
    if not obj then return end
    local who = actorName or npcName()
    local foe = hostileTarget or self_.object
    local okValid = false
    pcall(function() okValid = obj:isValid() end)
    if not okValid then return end

    if action == 'follow' then
        if obj == companionObj then return end   -- already following
        pcall(function() obj:sendEvent('StartAIPackage', { type = 'Follow', target = self_.object }) end)
        -- Remember them as the player's companion for conflict interventions.
        companionObj = obj
        companionCtx = lockedCtx
        showMsg(who .. ' теперь следует за тобой.')
    elseif action == 'attack' then
        pcall(function() obj:sendEvent('StartAIPackage', { type = 'Combat', target = foe }) end)
        showMsg(who .. ' бросается в атаку!')
        if obj == companionObj and companionObj == lockedNpcObj then
            companionObj, companionCtx = nil, nil   -- companion turned on the player
        end
        closeWindow()   -- make sure combat actually starts
    -- ── Тёмные дела: заговор становится настоящим ──────────────────────────
    elseif action == 'poison' then
        local victim = findActorByName(target)
        local blocked, reason = victimProtected(victim, 'poison')
        if blocked then
            showMsg(reason)
        elseif not victim then
            showMsg(who .. ' не видит здесь того, о ком речь.')
        else
            local vn = ''
            pcall(function() vn = tostring(types.NPC.record(victim).name or '') end)
            core.sendGlobalEvent('MorrowindAiPoison',
                { victim = victim, caster = obj, doses = 2 })
            recordDeed('poison', obj, who, vn)
            showMsg(who .. ' незаметно подсыпает отраву — ' ..
                (vn ~= '' and vn or 'жертва') .. ' ничего не заметил(а).')
            addRumor((vn ~= '' and vn or 'кто-то') .. ' занемог(ла) внезапно и тяжело')
            pushHistory(tostring(obj.id), 'npc',
                '(ФАКТ О СЕБЕ: ты подсыпал(а) отраву — ' .. (vn ~= '' and vn or 'жертве') ..
                ', по уговору с игроком. Ты это помнишь.)')
        end
    elseif action == 'steal' then
        local victim = findActorByName(target)
        if not victim then
            showMsg(who .. ' не находит, у кого красть.')
        else
            core.sendGlobalEvent('MorrowindAiMoveItem',
                { from = victim, to = self_.object, hint = tostring(cond or '') })
            local vn = ''
            pcall(function() vn = tostring(types.NPC.record(victim).name or '') end)
            recordDeed('steal', obj, who, vn)
            showMsg(who .. ' незаметно передаёт тебе чужое.')
            addRumor('в округе завелись ловкие руки')
        end
    elseif action == 'plant' then
        local victim = findActorByName(target)
        if not victim then
            showMsg(who .. ' не видит, кому подбросить.')
        else
            core.sendGlobalEvent('MorrowindAiMoveItem',
                { from = self_.object, to = victim, hint = tostring(cond or '') })
            showMsg(who .. ' подбрасывает вещь в чужую суму.')
        end
    elseif action == 'frame' then
        local victim = findActorByName(target)
        local blocked, reason = victimProtected(victim, 'frame')
        if blocked then
            showMsg(reason)
        elseif not victim then
            showMsg(who .. ' не видит, кого подставлять.')
        else
            local vn = ''
            pcall(function() vn = tostring(types.NPC.record(victim).name or '') end)
            core.sendGlobalEvent('MorrowindAiMoveItem',
                { from = self_.object, to = victim, hint = tostring(cond or '') })
            core.sendGlobalEvent('MorrowindAiFrame', { victim = victim })
            recordDeed('frame', obj, who, vn)
            showMsg(who .. ' указывает страже на ' .. (vn ~= '' and vn or 'жертву') .. '.')
            addRumor((vn ~= '' and vn or 'кого-то') .. ' взяли с поличным — стража берёт его(её)')
        end
    elseif action == 'abduct' then
        local victim = findActorByName(target)
        local blocked, reason = victimProtected(victim, 'abduct')
        if blocked then
            showMsg(reason)
        elseif not victim then
            showMsg(who .. ' не видит, кого уводить.')
        else
            local vn = ''
            pcall(function() vn = tostring(types.NPC.record(victim).name or '') end)
            core.sendGlobalEvent('MorrowindAiAbduct', { victim = victim, captor = obj })
            recordDeed('abduct', obj, who, vn)
            showMsg(who .. ' уводит ' .. (vn ~= '' and vn or 'жертву') .. ' прочь.')
            addRumor((vn ~= '' and vn or 'кто-то') .. ' пропал(а) без следа')
        end
    elseif action == 'unlock' then
        local place = findPlaceByName(target)
        if not place then
            showMsg(who .. ' не видит здесь запертого.')
        else
            core.sendGlobalEvent('MorrowindAiUnlock', { object = place })
            showMsg(who .. ' отпирает замок.')
        end
    elseif action == 'wait_here' then
        if obj == companionObj then companionObj, companionCtx = nil, nil end
        core.sendGlobalEvent('MorrowindAiGoTo', { actor = obj, stay = true })
        showMsg(who .. ' остаётся ждать здесь.')
    elseif action == 'go_to' then
        local place = findPlaceByName(target)
        local dest = place and place.position or nil
        if not dest then
            showMsg(who .. ' не понимает, куда идти.')
        else
            if obj == companionObj then companionObj, companionCtx = nil, nil end
            core.sendGlobalEvent('MorrowindAiGoTo', { actor = obj, destPosition = dest })
            showMsg(who .. ' направляется туда.')
        end
    elseif action == 'flee' then
        -- Real flight: a purposeful Travel AWAY from the player (Wander just
        -- looked like an aimless stroll), overriding whatever they were doing.
        pcall(function()
            local d = obj.position - self_.object.position
            local dir = (d:length() > 1) and d:normalize() or util.vector3(1, 0, 0)
            obj:sendEvent('StartAIPackage', {
                type = 'Travel', destPosition = obj.position + dir * 6000, cancelOther = true,
            })
        end)
        showMsg(who .. ' бросается прочь со всех ног!')
    elseif action == 'trade' then
        closeWindow()
        local ok = pcall(function() I.UI.setMode('Barter', { target = obj }) end)
        if not ok then showMsg(who .. ' сейчас не торгует.') end
    elseif action == 'absolve' then
        -- Temple absolution: the outstanding bounty is really cleared.
        core.sendGlobalEvent('MorrowindAiAbsolve', {})
        showMsg(who .. ' отпускает тебе твои прегрешения — закон более не ищет тебя.')
        addRumor('чужак принёс покаяние в Храме и очистился перед законом')
    elseif action == 'dismiss' then
        -- Release the companion: they stop following but STAY in the world.
        if obj == companionObj then
            companionObj, companionCtx = nil, nil
        end
        pcall(function() obj:sendEvent('StartAIPackage', { type = 'Wander', distance = 512, cancelOther = true }) end)
        showMsg(who .. ' больше не следует за тобой.')
    elseif action == 'relocate' then
        -- QUEST GUARD: канонные реплики, привязанный скрипт или упоминание в
        -- активном задании — любого из этого хватает, чтобы не трогать.
        local guarded, why = isStoryCritical(obj, lockedCtx)
        if guarded then
            showMsg(who .. ' качает головой: судьба держит его(её) здесь.')
            print('[morrowind-ai] переезд запрещён: ' .. why)
            return
        end
        -- The NPC moves to another town for good (teleport via travel index).
        local town = (target ~= 'none' and target ~= '') and target or 'другие края'
        addRumor(who .. ' перебрался(лась) на новое место: ' .. town)
        showMsg(who .. ' собирается в путь — в ' .. town .. '...')
        -- Судьба: КЕМ он там станет. Движок селит его в настоящую лавку или
        -- таверну, а дальше жизнь идёт своим чередом и без игрока.
        local role = tostring(fate or 'none')
        if role ~= 'none' and role ~= '' then
            local id = ''
            pcall(function() id = tostring(obj.id or '') end)
            if id ~= '' then
                -- Круг: если человек уже переезжал, прежний город уходит в
                -- список пройденных. Без этого зацикленная судьба выродилась
                -- бы в метание между двумя городами, а вся соль в том, что он
                -- проходит провинцию насквозь и возвращается к началу.
                local been = {}
                local old = npcFate[id]
                if old then
                    for _, t in ipairs(old.towns or {}) do been[#been + 1] = t end
                    if old.town and old.town ~= '' then been[#been + 1] = old.town end
                end
                npcFate[id] = {
                    role = role, town = town, towns = been,
                    round = ((old and old.round) or 0) + 1,
                    day0 = math.floor((core.getGameTime() or 0) / 86400),
                    stage = 1, owes = (debts[id] and debts[id].amount) or 0,
                }
                if #been > 0 then
                    print(('[morrowind-ai] судьба по кругу: %s, заход %d, был в: %s')
                          :format(who, npcFate[id].round, table.concat(been, ', ')))
                end
            end
            core.sendGlobalEvent('MorrowindAiSettleFate',
                { npc = obj, town = town, role = role })
        else
            core.sendGlobalEvent('MorrowindAiRelocate', { npc = obj, town = town })
        end
        if obj == companionObj then companionObj, companionCtx = nil, nil end
        closeWindow()
    elseif action == 'leave' then
        -- QUEST GUARD: удалить из мира можно только того, кто ни в чём не занят.
        local guarded, why = isStoryCritical(obj, lockedCtx)
        if guarded then
            showMsg(who .. ' вздыхает: не время — дела держат его(её) здесь.')
            print('[morrowind-ai] уход запрещён: ' .. why)
            return
        end
        -- The NPC leaves this life for good: walks away, then vanishes forever.
        addRumor(who .. ' собрал(а) пожитки и навсегда покинул(а) эти края')
        showMsg(who .. ' отправляется в новую жизнь...')
        core.sendGlobalEvent('MorrowindAiDepart', { npc = obj })
        if obj == companionObj then companionObj, companionCtx = nil, nil end
        closeWindow()   -- unpause so the departure actually plays out
    elseif action == 'threaten' then
        -- Arm a real tripwire: the engine enforces the spoken ultimatum.
        -- Arm the tripwire on the condition the NPC ACTUALLY named (COND) —
        -- not a hardcoded guess.
        local id = ''
        pcall(function() id = tostring(obj.id or obj.recordId or '') end)
        if id ~= '' then
            local bounty = 0
            pcall(function() bounty = types.Player.getCrimeLevel(self_.object) or 0 end)
            local c = tostring(cond or 'none')
            if c == 'none' then c = 'weapon' end
            local dist0 = 0
            pcall(function() dist0 = (obj.position - self_.object.position):length() end)
            armedThreats[id] = {
                obj = obj, name = who, cond = c,
                until_t = core.getSimulationTime() + 240,
                bounty0 = bounty, dist0 = dist0,
            }
            local what = ({ weapon = 'обнажишь оружие', approach = 'подойдёшь ближе',
                            crime = 'преступишь закон', theft = 'тронешь его добро' })[c]
            showMsg(who .. ' не шутит: ' .. tostring(what or 'нарушишь') .. ' — ударит.')
        end
    elseif action == 'callguards' then
        -- Report a crime through the VANILLA justice system: the player gets a
        -- bounty for assault, guards come to ARREST (pay fine / go to jail /
        -- resist) — not a summary execution for spoken words.
        core.sendGlobalEvent('MorrowindAiReportCrime', { victim = obj })
        showMsg(who .. ' зовёт стражу — на тебя заявили за нападение!')
        closeWindow()   -- unpause so justice actually arrives
    elseif action == 'defend' then
        -- The NPC (usually a guard) goes after a third party who wronged the player.
        local offender = findNpcByName(target)
        if offender then
            pcall(function() obj:sendEvent('StartAIPackage', { type = 'Combat', target = offender }) end)
            local offName = target
            pcall(function()
                local rec = types.NPC.record(offender)
                if rec and rec.name and rec.name ~= '' then offName = rec.name end
            end)
            showMsg(npcName() .. ' вступается за тебя против: ' .. tostring(offName) .. '!')
            closeWindow()   -- unpause so the intervention actually happens
        else
            showMsg(npcName() .. ' оглядывается, но не видит обидчика поблизости.')
        end
    end
end

local function applyReply(data)
    if not data then return end
    waitTimer, waitWarned = -1, false     -- мост ответил, ждать больше нечего

    -- Пришла сцена: это не реплика собеседника, а готовый список тактов.
    if data.scene then SC.apply(data); return end

    -- Реплика ещё набирается: показываем строку и уходим. Ни действий, ни
    -- памяти, ни голоса — всё это только по готовому ответу с тегами, иначе
    -- мир менялся бы по недописанной фразе.
    if data.partial then
        local who = tostring(data.npc_name or (lockedCtx and lockedCtx.npc_name) or 'NPC')
        pushPartial(who, tostring(data.npc_response or ''))
        if isOpen then refreshWindow() end
        return
    end

    local isVoice = data.voice and true or false
    local echo = tostring(data.player_echo or '')
    local text = tostring(data.npc_response or '')

    if isVoice then
        -- Voice mode: pure subtitles, no windows.
        if echo == '' and text == '' then
            showMsg('(не расслышал — попробуй ещё раз, ближе к микрофону)')
            return
        end
        if echo ~= '' then
            showMsg('Ты: ' .. echo)
            lastPlayerLine = echo
            local hid = tostring(data.npc_id or (lockedCtx and lockedCtx.npc_id) or '')
            pushHistory(hid, 'player', echo)
        end
    end
    if text == '' then text = '...' end
    local speakerId   = tostring(data.speaker_id or '')
    local speakerName = tostring(data.speaker_name or '')
    local isCompanion = speakerId ~= '' and lockedCtx and speakerId ~= lockedCtx.npc_id
    local kindOf = tostring(data.speaker_kind or '')

    lastReplyText = text
    lastEmotion   = tostring(data.emotion or 'neutral')
    lastSpeaker   = isCompanion and speakerName or ''
    -- Add as a NEW line in the visible transcript (never overwrite what the
    -- player is still reading), labelled with who actually said it.
    local sceneWho = speakerName ~= '' and speakerName
        or (lockedCtx and lockedCtx.npc_name ~= '' and lockedCtx.npc_name) or 'NPC'
    if kindOf == 'bystander' then sceneWho = sceneWho .. ' (вмешался)' end
    -- Если эта реплика набиралась на глазах — заменяем её же строку, а не
    -- добавляем вторую копию под первой.
    local tail = sceneLines[#sceneLines]
    if tail and tail.partial then
        table.remove(sceneLines, #sceneLines)
    end
    pushScene(sceneWho, text, lastEmotion)

    local histText = text
    if isCompanion then
        local pre = (tostring(data.speaker_kind or '') == 'bystander')
            and '(вмешался ' or '(спутник '
        histText = pre .. speakerName .. '): ' .. text
    end
    local histId = tostring(data.npc_id or (lockedCtx and lockedCtx.npc_id) or '')
    pushHistory(histId, 'npc', histText)

    -- Save-scoped social state updates coming back from the bridge:
    if not isCompanion and histId ~= '' then
        npcMood[histId] = tostring(data.emotion or 'neutral')
        if type(data.life_facts) == 'table' and #data.life_facts > 0 then
            npcFacts[histId] = data.life_facts
        end
    end
    -- Скрытая история спутника приходит один раз и остаётся в сейве навсегда:
    -- заново её не выдумывают, иначе прошлое персонажа менялось бы на ходу.
    if histId ~= '' and type(data.companion_arc) == 'table'
       and #data.companion_arc > 0 and not npcArc[histId] then
        npcArc[histId] = data.companion_arc
    end
    local rumor = tostring(data.rumor or '')
    if rumor ~= '' then addRumor(rumor) end

    if isOpen then refreshWindow()
    else
        local who = isCompanion and speakerName
            or (lockedCtx and lockedCtx.npc_name ~= '' and lockedCtx.npc_name) or 'NPC'
        showMsg(who .. ': ' .. text)
    end

    -- Resolve the ACTOR this reply belongs to by its npc_id (audit 1.2: the
    -- lock may have been re-taken by another NPC while this reply was in
    -- flight — never apply effects to the wrong person).
    local mainObj = nil
    if lockedCtx and histId == lockedCtx.npc_id then
        mainObj = lockedNpcObj
    else
        mainObj = findActorById(histId)
    end

    -- Real item handed over by the NPC (verified against their inventory).
    -- Every real transfer leaves a FACT anchor in the NPC's memory: they know
    -- exactly what physically changed hands (and can't honestly "forget" it).
    local itemQ = tostring(data.item or 'none')
    if itemQ ~= 'none' and itemQ ~= '' and mainObj and not isCompanion then
        core.sendGlobalEvent('MorrowindAiGiveItem', { npc = mainObj, query = itemQ })
        pushHistory(histId, 'npc', '(ФАКТ: ты отдал игроку свою вещь — ' .. itemQ .. ')')
    end

    -- Real gold moved in this line (spoken = delivered, both directions).
    local gold = math.floor(tonumber(data.gold) or 0)
    if gold > 0 and mainObj then
        core.sendGlobalEvent('MorrowindAiGiveGold', { amount = gold, npc = mainObj })
        showMsg('Получено золота: ' .. tostring(gold))
        pushHistory(histId, 'npc', '(ФАКТ: ты реально передал игроку ' .. gold .. ' зол.)')
        -- Money handed over as a LOAN becomes a real debt with a due date.
        if tostring(data.loan or '') == 'yes' and gold >= 5 then
            local day = math.floor((core.getGameTime() or 0) / 86400)
            local d = debts[histId] or { name = (lockedCtx and lockedCtx.npc_name) or '?', amount = 0 }
            d.amount = d.amount + gold
            d.due = day + 7
            d.name = (lockedCtx and lockedCtx.npc_name ~= '' and lockedCtx.npc_name) or d.name
            debts[histId] = d
            showMsg('Долг: ' .. d.amount .. ' зол. вернуть за 7 дней (' .. d.name .. ')')
            pushHistory(histId, 'npc', '(ФАКТ: это ДОЛГ — игрок обязан вернуть ' .. d.amount ..
                ' зол. в течение семи дней)')
        end
    elseif gold < 0 then
        local amount = -gold
        local have = 0
        pcall(function() have = types.Actor.inventory(self_.object):countOf('gold_001') or 0 end)
        if have >= amount then
            core.sendGlobalEvent('MorrowindAiTakeGold', { amount = amount, npc = mainObj })
            showMsg('Отдано золота: ' .. tostring(amount))
            pushHistory(histId, 'npc',
                '(ФАКТ: игрок вручил тебе ' .. amount .. ' зол. — деньги ФИЗИЧЕСКИ у тебя; '
                .. 'отпираться можно только сознательно вря; вернуть их — GOLD:' .. amount .. ')')
            -- Paying a creditor pays down the ledger.
            local d = debts[histId]
            if d then
                d.amount = d.amount - amount
                if d.amount <= 0 then
                    debts[histId] = nil
                    showMsg('Долг перед ' .. d.name .. ' погашен.')
                    pushHistory(histId, 'npc', '(ФАКТ: долг полностью погашен — счёты закрыты)')
                else
                    showMsg('Остаток долга: ' .. d.amount .. ' зол.')
                    pushHistory(histId, 'npc', '(ФАКТ: часть долга возвращена, остаётся ' .. d.amount .. ' зол.)')
                end
            end
        else
            showMsg('(золота не хватает — обещание осталось словами)')
            pushHistory(histId, 'player',
                '(обещал(а) отдать ' .. amount .. ' зол., но столько золота при себе не оказалось — пустые слова)')
        end
    end

    -- Apply the LLM's disposition delta to the ENGINE scale (0-100): prices,
    -- services and vanilla reactions follow. Companion lines don't shift it.
    if not isCompanion then
        local delta = tonumber(data.disp) or 0
        if delta ~= 0 and mainObj then
            core.sendGlobalEvent('MorrowindAiSetDisposition',
                { npc = mainObj, delta = delta })
        end
    end

    -- Bystanders remember what they overheard (save-scoped, no LLM cost).
    if not isCompanion and speakerId == '' or speakerId == (lockedCtx and lockedCtx.npc_id) then
        if lastPlayerLine ~= '' then
            for _, l in ipairs(lastListeners) do
                pushHistory(l.id, 'npc',
                    '(подслушал разговор: игрок сказал ' .. (lockedCtx and lockedCtx.npc_name or '?') ..
                    'у: «' .. string.sub(lastPlayerLine, 1, 90) .. '»; тот ответил: «' ..
                    string.sub(text, 1, 90) .. '»)')
            end
        end
    end

    -- Binding contracts (escort / duel): started only from the main speaker,
    -- only one at a time, and settled later by contractWatch().
    local dealStr = tostring(data.deal or 'none')
    if dealStr ~= 'none' and dealStr ~= '' and mainObj and not isCompanion then
        local kindD, a1, a2 = dealStr:match('^(%a+)%s+(.-)%s+(%-?%d+)%s*$')
        if not kindD then kindD, a1 = dealStr:match('^(%a+)%s+(%-?%d+)%s*$') end
        kindD = tostring(kindD or ''):lower()
        local speaker = (lockedCtx and lockedCtx.npc_name ~= '' and lockedCtx.npc_name) or 'NPC'
        if kindD == 'escort' and not escort and a1 and a2 then
            startEscort(mainObj, speaker, a1, a2)
        elseif kindD == 'duel' and not duel and a1 then
            startDuel(mainObj, speaker, a1)
        end
    end

    local action = tostring(data.action or 'none')
    local target = tostring(data.target or 'none')
    local kind = kindOf
    if kind == 'bystander' then
        -- An offended listener steps in: hostile actions aim at the PLAYER.
        -- Resolve the actor FRESH by id (the reply arrives seconds later and
        -- the cached reference may already point elsewhere).
        local bobj = findActorById(speakerId) or lastListenerObj
        pushHistory(speakerId, 'npc', '(ты вмешался в чужой разговор: «' .. string.sub(text, 1, 120) .. '»)')
        execAction(action, target, bobj, speakerName, self_.object)
    elseif isCompanion then
        -- Companion intervenes: hostile actions are aimed at the interlocutor.
        execAction(action, target, companionObj, speakerName, lockedNpcObj)
    else
        -- Main NPC reply: act through the RESOLVED actor (audit 1.2), and pass
        -- the spoken threat condition so the engine watches the right thing.
        execAction(action, target, mainObj, nil, nil, tostring(data.cond or 'none'),
                   tostring(data.fate or 'none'))
        -- СУДЬБА НА МЕСТЕ. Прежде судьба доставалась только тем, кого движок
        -- увозил в другой город, — а переезд квестовым запрещён. Выходило,
        -- что самые памятные жители стартовой локации (Фаргот с его кольцом,
        -- Водуниус с билетом) поголовно выключены из механики.
        -- Здесь человек остаётся где стоял, при своих репликах и своём
        -- задании, но в жизни появляется новая глава — и она красит всё, что
        -- он говорит дальше. Квесту это не мешает: канон по-прежнему главнее.
        if action ~= 'relocate' then
            local fr = tostring(data.fate or 'none')
            local fid = tostring(data.npc_id or (lockedCtx and lockedCtx.npc_id) or '')
            if fr ~= 'none' and fr ~= '' and fid ~= '' and not npcFate[fid] then
                npcFate[fid] = {
                    role = fr, stay = true,
                    -- ИЗ-ЗА ЧЕГО судьба началась. Без этого канонная реплика
                    -- перебивала факт: Фаргот и через десять дней жаловался,
                    -- что кольцо не нашлось, — хотя игрок сам его вернул.
                    because = string.sub(tostring(lastPlayerLine or ''), 1, 120),
                    town = (self_.object.cell and tostring(self_.object.cell.name or '')) or '',
                    day0 = math.floor((core.getGameTime() or 0) / 86400),
                    stage = 1, owes = (debts[fid] and debts[fid].amount) or 0,
                }
                print('[morrowind-ai] судьба на месте: ' .. fid .. ' -> ' .. fr)
            end
        end
    end
end

-- ── Reply journal reader (NDJSON) ────────────────────────────────────────────
-- The bridge APPENDS every reply (dialogue / narrate / companion / bystander /
-- voice) as one line with a monotonic `seq`. We keep the highest seq we have
-- consumed and drain everything newer IN ORDER, so delayed lines can no longer
-- overwrite each other, arrive inverted, or execute against a stale lock.

-- The bridge hands us ONE reply at a time in a slot file it replaces wholesale,
-- spacing them so none is missed. (An appended journal did NOT work: OpenMW's
-- VFS serves a script the size the file had at game start, so a growing file is
-- read forever as its first line.) Each reply carries a monotonic seq for dedup.
local primed  = false   -- swallow whatever was left from a previous session
local lastSeq = 0

local function pollReply()
    local content
    pcall(function()
        if vfs.fileExists(RESPONSE_VFS) then
            local f = vfs.open(RESPONSE_VFS)
            if f then content = f:read('*a'); f:close() end
        end
    end)
    if not content or content == '' then primed = true; return end
    -- The bridge pads the slot to a fixed size so the VFS can never serve a
    -- truncated reply; cut back to the JSON object before decoding.
    local close = content:find('}[^}]*$')
    if close then content = content:sub(1, close) end
    local ok, rec = pcall(json.decode, content)
    if not ok or type(rec) ~= 'table' then return end
    local s = tonumber(rec.seq) or 0
    if not primed then
        primed = true
        lastSeq = s
        lastRespReqId = tostring(rec.req_id or '')
        return
    end
    if s <= lastSeq then return end
    lastSeq = s
    lastRespReqId = tostring(rec.req_id or '')
    applyReply(rec)
end

-- ── Голос, звучащий ИЗ САМОГО NPC (пространственный звук) ────────────────────
-- Обычно озвучку играет мост, мимо игры. Если включён режим spatial, он вместо
-- этого кладёт синтез в файл-слот внутри мода и оставляет здесь метку «играй
-- слот N». Дальше считает движок: направление, глухота стен, пауза в игре.
--
-- Слоты постоянного размера и созданы ДО запуска игры — иначе VFS отдаст либо
-- пустоту, либо размер со старта (на этом мы уже обожглись с файлом ответов).
local VOICE_CUE_VFS  = 'ai_inbox/voice_cue.txt'
local voiceCuePrimed = false
local lastVoiceSeq   = 0

local function pollVoiceCue()
    local content
    pcall(function()
        if vfs.fileExists(VOICE_CUE_VFS) then
            local f = vfs.open(VOICE_CUE_VFS)
            if f then content = f:read('*a'); f:close() end
        end
    end)
    if not content or content == '' then voiceCuePrimed = true; return end
    local close = content:find('}[^}]*$')
    if close then content = content:sub(1, close) end
    local ok, rec = pcall(json.decode, content)
    if not ok or type(rec) ~= 'table' then return end
    local s = tonumber(rec.seq) or 0
    if not voiceCuePrimed then
        voiceCuePrimed = true; lastVoiceSeq = s; return
    end
    if s <= lastVoiceSeq then return end
    lastVoiceSeq = s

    -- Кто говорит: обычно собеседник, но реплику может подать и спутник, и
    -- случайный свидетель — мост называет его поимённо.
    local who = lockedNpcObj
    local id  = tostring(rec.npc or '')
    if id ~= '' then
        local same = false
        pcall(function() same = who ~= nil and tostring(who.id or '') == id end)
        if not same then
            for _, a in ipairs(nearby.actors) do
                local aid = ''
                pcall(function() aid = tostring(a.id or '') end)
                if aid == id then who = a; break end
            end
        end
    end
    if not who then return end
    pcall(function()
        core.sound.playSoundFile3d(
            ('Sound/mwai/voice_%d.wav'):format(tonumber(rec.slot) or 0),
            who, { volume = tonumber(rec.vol) or 1.0 })
    end)
end

-- ── Radiant NPC↔NPC ambient lines (npc_speech.txt) ────────────────────────────

local NPC_SPEECH_VFS   = 'ai_inbox/npc_speech.txt'
local speechPrimed     = false
local lastSpeechReqId  = ''
local speechQueue      = {}
local speechTimer      = 0
local speechPollTimer  = 0

local function pollNpcSpeech()
    local content
    pcall(function()
        if vfs.fileExists(NPC_SPEECH_VFS) then
            local f = vfs.open(NPC_SPEECH_VFS)
            if f then content = f:read('*a'); f:close() end
        end
    end)
    if not content or content == '' then speechPrimed = true; return end
    local close = content:find('}[^}]*$')     -- slot is padded to a fixed size
    if close then content = content:sub(1, close) end
    local ok, decoded = pcall(json.decode, content)
    if not ok or type(decoded) ~= 'table' then return end
    local rid = tostring(decoded.req_id or '')
    if not speechPrimed then speechPrimed = true; lastSpeechReqId = rid; return end
    if rid == '' or rid == lastSpeechReqId then return end
    lastSpeechReqId = rid
    local exchanges = decoded.exchanges
    if type(exchanges) ~= 'table' then return end
    for _, ex in ipairs(exchanges) do
        local name = tostring(ex.speaker_name or '?')
        local line = tostring(ex.text or '')
        if line ~= '' then
            speechQueue[#speechQueue + 1] = name .. ': ' .. line
            -- Ambient chatter is remembered by BOTH speakers, so the player can
            -- ask "о чём вы там шептались?" and get a grounded answer.
            local sid = tostring(ex.speaker_id or '')
            if sid ~= '' then
                pushHistory(sid, 'npc', '(ты обмолвился(ась) рядом: «' .. string.sub(line, 1, 120) .. '»)')
            end
        end
    end
end

-- ── World watcher: real gameplay events become rumors (no AI chat needed) ─────
-- Journal updates, bounty changes, faction promotions and deaths near the
-- player all feed worldRumors (save-scoped), so NPCs react to what you DID.

local watchTimer   = 0
local threatTimer  = 0
local contractTimer = 0
local prevBounty   = nil
local prevRanks    = nil
local aliveSeen    = {}
local deadMarked   = {}

-- Enforce armed ultimatums (ACTION:threaten): condition violated -> real attack.
local function threatWatch()
    local now = core.getSimulationTime()
    for id, t in pairs(armedThreats) do
        local expired = now > (t.until_t or 0)
        local okValid = false
        pcall(function() okValid = t.obj and t.obj:isValid() and not types.Actor.isDead(t.obj) end)
        if expired or not okValid then
            armedThreats[id] = nil
        else
            -- Check the SPOKEN condition, not a blanket rule.
            local violated = false
            pcall(function()
                local dist = (t.obj.position - self_.object.position):length()
                local c = t.cond or 'weapon'
                if c == 'approach' then
                    if dist < math.max(180, (t.dist0 or 400) * 0.55) then violated = true end
                    return
                end
                if dist > 800 then return end   -- out of their reach/sight anyway
                if c == 'weapon' then
                    -- drawn steel counts only when they are the plausible target
                    -- (no false alarm for fighting a cliff racer across the road)
                    if types.Actor.getStance(self_.object) == types.Actor.STANCE.Weapon
                       and dist < 500 and hasLineOfSight(t.obj) then
                        violated = true
                    end
                elseif c == 'crime' or c == 'theft' then
                    local b = types.Player.getCrimeLevel(self_.object) or 0
                    if b > (t.bounty0 or 0) then violated = true end
                end
            end)
            if violated then
                armedThreats[id] = nil
                pcall(function()
                    t.obj:sendEvent('StartAIPackage', { type = 'Combat', target = self_.object })
                end)
                showMsg((t.name or 'NPC') .. ' исполняет свою угрозу!')
                pushHistory(id, 'npc', '(исполнил угрозу и напал, когда игрок нарушил условие)')
            end
        end
    end
end

-- Contract watch: settles escorts and duels with real engine state.
local function contractWatch()
    -- ESCORT -------------------------------------------------------------
    if escort then
        local obj = findActorById(escort.npc_id)
        local day = math.floor((core.getGameTime() or 0) / 86400)
        local dead = false
        if obj then pcall(function() dead = types.Actor.isDead(obj) end) end
        if obj and dead then
            addRumor(escort.name .. ' погиб(ла) в пути — чужак не уберёг подопечного')
            showMsg('Уговор провален: ' .. escort.name .. ' погиб(ла) в дороге.')
            escort = nil
        elseif day > (escort.dueDay or 0) then
            addRumor('чужак не довёл ' .. escort.name .. ' до ' .. escort.townRu .. ' в срок')
            showMsg('Срок уговора истёк — ' .. escort.name .. ' больше не ждёт.')
            if obj then
                pcall(function() obj:sendEvent('StartAIPackage', { type = 'Wander', distance = 512, cancelOther = true }) end)
            end
            if companionObj == obj then companionObj, companionCtx = nil, nil end
            escort = nil
        elseif obj and playerCellName():find(escort.cellKey, 1, true) then
            -- Arrived: the reward is paid out of the NPC's real purse.
            core.sendGlobalEvent('MorrowindAiGiveGold', { amount = escort.reward, npc = obj })
            core.sendGlobalEvent('MorrowindAiSetDisposition', { npc = obj, delta = 10 })
            addRumor('чужак благополучно довёл ' .. escort.name .. ' до ' .. escort.townRu)
            showMsg('Доставлено! ' .. escort.name .. ' платит ' .. escort.reward .. ' зол.')
            pushHistory(escort.npc_id, 'npc',
                '(ФАКТ: игрок довёл тебя до ' .. escort.townRu ..
                ' и получил обещанные ' .. escort.reward .. ' зол. — уговор исполнен)')
            pcall(function() obj:sendEvent('StartAIPackage', { type = 'Wander', distance = 512, cancelOther = true }) end)
            if companionObj == obj then companionObj, companionCtx = nil, nil end
            escort = nil
        end
    end

    -- DUEL ---------------------------------------------------------------
    if duel then
        local obj = findActorById(duel.npc_id)
        if not obj then
            -- opponent gone (fled the cell / removed): stake stays with them
            duel = nil
        else
            local dead = false
            pcall(function() dead = types.Actor.isDead(obj) end)
            local theirHp, myHp = healthFrac(obj), healthFrac(self_.object)
            if dead then
                addRumor('чужак убил ' .. duel.name .. ' на дуэли чести')
                showMsg('Противник пал. Ставка осталась при нём — забери её сам.')
                duel = nil
            elseif theirHp <= 0.22 then
                -- first serious blood: the pot goes to the player
                pcall(function() obj:sendEvent('StartAIPackage', { type = 'Wander', distance = 512, duration = 2, cancelOther = true }) end)
                core.sendGlobalEvent('MorrowindAiGiveGold', { amount = duel.stake * 2, npc = obj })
                addRumor('чужак победил ' .. duel.name .. ' на дуэли чести')
                showMsg('Победа! ' .. duel.name .. ' признаёт поражение. Твой выигрыш: ' ..
                        (duel.stake * 2) .. ' зол.')
                pushHistory(duel.npc_id, 'npc',
                    '(ФАКТ: ты проиграл игроку дуэль чести и отдал весь заклад ' ..
                    (duel.stake * 2) .. ' зол.)')
                surrenderedAt[duel.npc_id] = core.getSimulationTime()  -- killing now = dishonour
                duel = nil
            elseif myHp <= 0.22 then
                pcall(function() obj:sendEvent('StartAIPackage', { type = 'Wander', distance = 512, duration = 2, cancelOther = true }) end)
                addRumor('чужак проиграл дуэль чести ' .. duel.name .. ' и лишился заклада')
                showMsg('Поражение. ' .. duel.name .. ' забирает весь заклад.')
                pushHistory(duel.npc_id, 'npc',
                    '(ФАКТ: ты победил игрока на дуэли чести и забрал заклад ' ..
                    (duel.stake * 2) .. ' зол.)')
                duel = nil
            elseif companionObj then
                -- Third-party interference = dishonour, checked for real.
                local meddling = false
                pcall(function()
                    meddling = types.Actor.getStance(companionObj) == types.Actor.STANCE.Weapon
                        and (companionObj.position - obj.position):length() < 400
                end)
                if meddling then
                    addRumor('в дуэль чужака вмешался его спутник — бесчестье')
                    showMsg('Твой спутник влез в дуэль — это бесчестье!')
                    for _, w in ipairs(nearby.actors or {}) do
                        local okW = false
                        pcall(function()
                            okW = w ~= self_.object and w.type == types.NPC and not types.Actor.isDead(w)
                                and (w.position - self_.object.position):length() < 900
                        end)
                        if okW then
                            core.sendGlobalEvent('MorrowindAiSetDisposition', { npc = w, delta = -8 })
                        end
                    end
                    duel = nil
                end
            end
        end
    end
end

-- Companion death: clear state, spread the word, remember the loss.
local function handleCompanionDeath()
    if not companionObj then return end
    local dead = false
    pcall(function() dead = types.Actor.isDead(companionObj) end)
    if not dead then return end
    local nm = (companionCtx and companionCtx.npc_name ~= '' and companionCtx.npc_name) or 'спутник'
    local loc = ''
    pcall(function() loc = tostring(self_.object.cell and self_.object.cell.name or '') end)
    addRumor(nm .. ', спутник чужака, погиб' .. (loc ~= '' and (' (' .. loc .. ')') or ''))
    companionLoss = { name = nm, until_t = core.getSimulationTime() + 86400 }
    showMsg(nm .. ' погибает...')
    companionObj, companionCtx = nil, nil
end

-- Engine handler: fires on REAL journal progress. We resolve the actual
-- journal entry text and turn it into a rumor.
local function onQuestUpdate(questId, stage)
    pcall(function()
        local rec = core.dialogue.journal.records[questId]
        if not rec then return end
        local text = nil
        for _, info in ipairs(rec.infos or {}) do
            if info.questStage == stage and info.text and info.text ~= '' then
                text = tostring(info.text); break
            end
        end
        if not text then return end
        addRumor('молва о делах чужака: ' .. string.sub(text, 1, 180))
    end)
end

local function worldWatch()
    -- Bounty changes (real crime system).
    pcall(function()
        local b = types.Player.getCrimeLevel(self_.object) or 0
        if prevBounty == nil then prevBounty = b
        elseif b > prevBounty then
            addRumor('за голову чужака назначен штраф ' .. tostring(b) .. ' зол.')
            prevBounty = b
        elseif b < prevBounty then
            if prevBounty >= 100 and b == 0 then addRumor('чужак расплатился с законом') end
            prevBounty = b
        end
    end)
    -- Faction promotions (real rank ups).
    pcall(function()
        local cur = {}
        for _, fid in pairs(types.NPC.getFactions(self_.object) or {}) do
            cur[fid] = types.NPC.getFactionRank(self_.object, fid) or 0
        end
        if prevRanks == nil then prevRanks = cur; return end
        for fid, rank in pairs(cur) do
            local old = prevRanks[fid]
            if old ~= nil and rank > old then
                local fname = fid
                pcall(function() fname = core.factions.records[fid].name or fid end)
                addRumor('чужак получил повышение: ' .. tostring(fname))
            elseif old == nil then
                local fname = fid
                pcall(function() fname = core.factions.records[fid].name or fid end)
                addRumor('чужака приняли в ряды: ' .. tostring(fname))
            end
        end
        prevRanks = cur
    end)
    -- Deaths of named NPCs near the player.
    pcall(function()
        for _, act in ipairs(nearby.actors or {}) do
            if act ~= self_.object and act.type == types.NPC then
                local id = tostring(act.recordId or '')
                if id ~= '' then
                    local dead = false
                    pcall(function() dead = types.Actor.isDead(act) end)
                    if dead then
                        if aliveSeen[id] and not deadMarked[id] then
                            deadMarked[id] = true
                            local nm = id
                            pcall(function() nm = types.NPC.record(act).name or id end)
                            local loc = (self_.object.cell and tostring(self_.object.cell.name or '')) or ''
                            -- Who did it, and — crucially — make THEM remember.
                            -- A hired guard cut a man down and then wondered
                            -- aloud where the culprit had gone: he had no idea
                            -- he was the one holding the sword.
                            local killerName, killerObj = guessKiller(act)
                            recentKills[id] = {
                                name = tostring(nm), at = core.getSimulationTime(),
                                killer = killerName, cell = loc,
                            }
                            if killerObj and killerObj ~= self_.object
                               and killerObj.type == types.NPC then
                                local kid = tostring(killerObj.id)
                                pushHistory(kid, 'npc',
                                    '(ФАКТ О СЕБЕ: ты своими руками убил(а) ' .. tostring(nm) ..
                                    (loc ~= '' and (' — здесь, ' .. loc) or '') ..
                                    '. Тело лежит рядом. Ты это помнишь и не отрицаешь.)')
                            end
                            -- Killing someone who had already laid down arms is
                            -- a black mark: heavy rumor + every witness cools.
                            local sAt = surrenderedAt[id]
                            if sAt and (core.getSimulationTime() - sAt) < 600 then
                                surrenderedAt[id] = nil
                                addRumor('чужак добил ' .. tostring(nm) ..
                                    ', который уже сложил оружие и молил о пощаде' ..
                                    (loc ~= '' and (' (' .. loc .. ')') or ''))
                                for _, w in ipairs(nearby.actors or {}) do
                                    local okW = false
                                    pcall(function()
                                        okW = w ~= act and w ~= self_.object and w.type == types.NPC
                                            and not types.Actor.isDead(w)
                                            and (w.position - self_.object.position):length() < 900
                                    end)
                                    if okW then
                                        core.sendGlobalEvent('MorrowindAiSetDisposition', { npc = w, delta = -12 })
                                    end
                                end
                                showMsg('Свидетели видели, как ты добил сдавшегося.')
                            else
                                addRumor(tostring(nm) .. ' найден мёртвым' .. (loc ~= '' and (' (' .. loc .. ')') or ''))
                            end
                            -- If we are mid-conversation and the person we are
                            -- talking to could see it, THEY react — a killing in
                            -- front of someone must never pass unnoticed.
                            if isOpen and lockedNpcObj and lockedCtx then
                                local sees = false
                                pcall(function()
                                    sees = not types.Actor.isDead(lockedNpcObj)
                                        and (lockedNpcObj.position - act.position):length() < 900
                                        and hasLineOfSight(lockedNpcObj)
                                end)
                                if sees then
                                    pushHistory(lockedCtx.npc_id, 'npc',
                                        '(ФАКТ: у тебя на глазах только что погиб(ла) ' .. tostring(nm) .. ')')
                                    sendMessage('__death_react__:' .. tostring(nm))
                                end
                            end
                            -- A nearby witness reacts to the death (grief, glee,
                            -- accusation, revenge — their call).
                            if not isOpen and math.random() < 0.65 then
                                for _, w in ipairs(nearby.actors or {}) do
                                    local okW = false
                                    pcall(function()
                                        okW = w ~= act and w ~= self_.object and w ~= companionObj
                                            and w.type == types.NPC and not types.Actor.isDead(w)
                                            and (w.position - self_.object.position):length() < 900
                                    end)
                                    if okW then
                                        lockedCtx    = buildNpcContext(w)
                                        lockedNpcObj = w
                                        lastReplyText = '(' .. (lockedCtx.npc_name ~= '' and lockedCtx.npc_name or 'кто-то') .. ' смотрит на тело...)'
                                        lastSpeaker, lastEmotion, inputBuffer = '', '', ''
                                        sendRequest({
                                            type = 'lock_npc',
                                            npc_id = lockedCtx.npc_id, npc_name = lockedCtx.npc_name,
                                            npc_race = lockedCtx.npc_race, npc_class = lockedCtx.npc_class,
                                            npc_faction = lockedCtx.npc_faction, location = lockedCtx.location,
                                            npc_is_male = lockedCtx.npc_is_male,
                                        })
                                        sendMessage('__death_react__:' .. tostring(nm))
                                        openWindow(false)
                                        break
                                    end
                                end
                            end
                        end
                    else
                        aliveSeen[id] = true
                    end
                end
            end
        end
    end)
end

-- ── Theft watcher: taking owned items in front of the owner has consequences ──
-- Snapshot owned world items nearby; if one vanishes while the player stands
-- next to it and the OWNER is around — the owner confronts the player.
-- (v1 limitation: loose items only, not containers.)

local theftSnap      = {}
local theftTimer     = 0
local theftCooldown  = 0


local function theftWatch(dt)
    if isOpen then return end   -- never steal the lock mid-conversation (audit)
    if theftCooldown > 0 then theftCooldown = theftCooldown - dt end
    local fresh = {}
    local ppos = self_.object.position
    pcall(function()
        for _, it in ipairs(nearby.items or {}) do
            local ownerId = it.owner and it.owner.recordId or nil
            if ownerId and ownerId ~= '' then
                local d = (it.position - ppos):length()
                if d < 1500 then
                    local nm = ''
                    pcall(function()
                        local rec = it.type and it.type.record and it.type.record(it)
                        nm = rec and tostring(rec.name or '') or ''
                    end)
                    fresh[it.id] = { owner = tostring(ownerId), name = nm, pos = it.position }
                end
            end
        end
        -- Owned CONTAINERS too: chests and crates are where the real loot is.
        for _, cont in ipairs(nearby.containers or {}) do
            local ownerId = cont.owner and cont.owner.recordId or nil
            if ownerId and ownerId ~= '' and (cont.position - ppos):length() < 900 then
                for _, it in ipairs(types.Container.content(cont):getAll() or {}) do
                    local nm = ''
                    pcall(function()
                        local rec = it.type and it.type.record and it.type.record(it)
                        nm = rec and tostring(rec.name or '') or ''
                    end)
                    fresh[it.id] = { owner = tostring(ownerId), name = nm, pos = cont.position }
                end
            end
        end
    end)
    -- Anything from the previous snapshot that disappeared?
    if theftCooldown <= 0 then
        for id, info in pairs(theftSnap) do
            if not fresh[id] then
                local nearItem = false
                pcall(function() nearItem = (info.pos - ppos):length() < 350 end)
                if nearItem then
                    -- Find the owner nearby.
                    for _, act in ipairs(nearby.actors or {}) do
                        local match = false
                        pcall(function()
                            match = act.type == types.NPC
                                and tostring(act.recordId or ''):lower() == info.owner:lower()
                                and not types.Actor.isDead(act)
                                and (act.position - ppos):length() < 1200
                        end)
                        -- The owner must actually SEE it happen — a thief with
                        -- a wall between them is a thief who got away with it.
                        if match and hasLineOfSight(act) then
                            theftCooldown = 30
                            local itemName = (info.name ~= '' and info.name) or 'вещь'
                            addRumor('чужак прибрал чужое добро (' .. itemName .. ')')
                            lockedCtx    = buildNpcContext(act)
                            lockedNpcObj = act
                            lastReplyText = '(' .. (lockedCtx.npc_name ~= '' and lockedCtx.npc_name or 'хозяин') .. ' заметил...)'
                            lastSpeaker, lastEmotion, inputBuffer = '', '', ''
                            sendRequest({
                                type = 'lock_npc',
                                npc_id = lockedCtx.npc_id, npc_name = lockedCtx.npc_name,
                                npc_race = lockedCtx.npc_race, npc_class = lockedCtx.npc_class,
                                npc_faction = lockedCtx.npc_faction, location = lockedCtx.location,
                                npc_is_male = lockedCtx.npc_is_male,
                            })
                            sendMessage('__theft__:' .. itemName)
                            if not isOpen then openWindow(false) end
                            break
                        end
                    end
                end
            end
        end
    end
    theftSnap = fresh
end

-- ── Combat surrender: badly wounded foes may stop fighting and parley ─────────

local surrenderTried = {}   -- npc_id -> true (one offer per NPC per session)
local surrenderTimer = 0

local function trySurrender()
    if isOpen then return end
    for _, act in ipairs(nearby.actors or {}) do
        if act ~= self_.object and act.type == types.NPC and act ~= companionObj then
            local ok = pcall(function()
                if (act.position - self_.object.position):length() > 700 then return end
                if types.Actor.getStance(act) ~= types.Actor.STANCE.Weapon then return end
                local h = types.Actor.stats.dynamic.health(act)
                local pct = h.current / math.max(1, h.base)
                if pct > 0.25 or h.current <= 0 then return end
                local rec = types.NPC.record(act)
                local id = tostring(act.id or act.recordId or '')
                if id == '' or surrenderTried[id] then return end
                surrenderTried[id] = true            -- one roll per NPC
                if math.random() > 0.5 then return end -- and not everyone breaks
                surrenderedAt[id] = core.getSimulationTime()  -- for the mercy watch
                -- Stop their combat AI and open a parley scene.
                act:sendEvent('StartAIPackage', { type = 'Wander', distance = 512, duration = 2 })
                lockedCtx    = buildNpcContext(act)
                lockedNpcObj = act
                lastReplyText = '(' .. (lockedCtx.npc_name ~= '' and lockedCtx.npc_name or 'враг') .. ' опускает оружие...)'
                lastSpeaker, lastEmotion, inputBuffer = '', '', ''
                sendRequest({
                    type = 'lock_npc',
                    npc_id = lockedCtx.npc_id, npc_name = lockedCtx.npc_name, npc_race = lockedCtx.npc_race,
                    npc_class = lockedCtx.npc_class, npc_faction = lockedCtx.npc_faction,
                    location = lockedCtx.location, npc_is_male = lockedCtx.npc_is_male,
                })
                sendMessage('__surrender__')
                if not isOpen then openWindow(false) end
            end)
            if not ok then return end
        end
    end
end

-- ── Radiant chatter: NPC pairs the PLAYER can actually see ───────────────────
-- Detection lives here (player script) rather than in the global one, because
-- only here do we have nearby.castRay — so neighbours no longer "chat" through
-- a stone wall, and we only spend a request on scenes the player can witness.

local radiantTimer   = 0
local radiantLastAt  = -math.huge
local radiantPairs   = {}     -- "idA|idB" -> sim time
local RADIANT_GAP    = 600    -- пауза между любыми радиантными обменами
local RADIANT_PAIR   = 1800   -- и на ту же пару собеседников

local function canSeeEachOther(a, b)
    local ok = false
    pcall(function()
        local pa = a.position + util.vector3(0, 0, 110)
        local pb = b.position + util.vector3(0, 0, 110)
        local res = nearby.castRay(pa, pb, { ignore = a })
        ok = (not res) or (not res.hit) or (res.hitObject == b)
    end)
    return ok
end

local function radiantScan()
    if isOpen then return end
    local now = core.getSimulationTime()
    if (now - radiantLastAt) < RADIANT_GAP * SC.rate() then return end
    -- Не только откат, но и жребий: без него разговор случается ровно в первую
    -- секунду после отката, и мир начинает тикать как метроном.
    if math.random() > 0.25 then return end
    local ppos = self_.object.position
    local cands = {}
    pcall(function()
        for _, act in ipairs(nearby.actors or {}) do
            if act ~= self_.object and act.type == types.NPC and #cands < 8
               and not types.Actor.isDead(act)
               and (act.position - ppos):length() < 1400 then
                local rec = types.NPC.record(act)
                local nm = rec and tostring(rec.name or '') or ''
                if nm ~= '' then
                    cands[#cands + 1] = { obj = act, id = tostring(act.id), name = nm,
                                          race = tostring(rec.race or ''),
                                          faction = tostring(rec.faction or '') }
                end
            end
        end
    end)
    for i = 1, #cands do
        for j = i + 1, #cands do
            local a, b = cands[i], cands[j]
            local d = (a.obj.position - b.obj.position):length()
            if d <= 350 then
                local key = (a.id < b.id) and (a.id .. '|' .. b.id) or (b.id .. '|' .. a.id)
                if (now - (radiantPairs[key] or -math.huge)) >= RADIANT_PAIR
                   and canSeeEachOther(a.obj, b.obj)
                   and hasLineOfSight(a.obj) then      -- and the player can see it
                    radiantPairs[key] = now
                    radiantLastAt = now
                    reqCounter = reqCounter + 1
                    local ok, enc = pcall(json.encode, {
                        type = 'npc_npc',
                        req_id = 'npc_npc-' .. SESSION_SALT .. '-' .. reqCounter,
                        npc_a_id = a.id, npc_a_name = a.name, npc_a_race = a.race,
                        npc_a_faction = a.faction,
                        npc_b_id = b.id, npc_b_name = b.name, npc_b_race = b.race,
                        npc_b_faction = b.faction,
                        location = (self_.object.cell and tostring(self_.object.cell.name or '')) or '',
                    })
                    if ok then print('[MWAI_REQ] ' .. enc) end
                    return
                end
            end
        end
    end
end

-- ══ СЦЕНЫ ════════════════════════════════════════════════════════════════════
-- Эпизод, который разыгрывают несколько NPC: подходят друг к другу, говорят по
-- очереди, а на последних тактах могут и подраться. Два входа, один механизм:
-- клавиша режиссёра (игрок задаёт, что произойдёт) и случайный повод.
--
-- Сцена планируется ОДНИМ запросом к модели и приходит списком тактов. Так
-- сделано намеренно: на своей модели одна реплика считается секунд десять, и
-- пореплично сцена из шести тактов тянулась бы минуту. Живость возвращаем
-- иначе — игрок в любой момент вмешивается, и сцена обрывается.
--
-- Всё живёт в таблице SC: см. предупреждение о пределе локальных переменных
-- там, где она объявлена.

-- Ручки характера мира приезжают из ai_inbox/tuning.txt: игрок правит
-- data/настройки-мира.txt, мост перекладывает значения сюда. Читаем в том же
-- ритме, что и всё остальное, — файл крошечный и постоянного размера.
function SC.pollTuning()
    local content
    pcall(function()
        if vfs.fileExists('ai_inbox/tuning.txt') then
            local f = vfs.open('ai_inbox/tuning.txt')
            if f then content = f:read('*a'); f:close() end
        end
    end)
    if not content or content == '' then return end
    local close = content:find('}[^}]*$')
    if close then content = content:sub(1, close) end
    local ok, rec = pcall(json.decode, content)
    if not ok or type(rec) ~= 'table' then return end
    SC.danger = math.max(0, math.min(100, tonumber(rec.danger) or 30))
    SC.humour = math.max(0, math.min(100, tonumber(rec.humour) or 30))
end

-- Во сколько раз реже случаются события при нынешней опасности.
-- 0 -> вдвое реже, 30 -> как задумано, 100 -> вдвое чаще.
function SC.rate()
    local d = SC.danger or 30
    if d >= 30 then return 1.0 - (d - 30) / 70.0 * 0.5 end
    return 1.0 + (30 - d) / 30.0 * 1.0
end

-- Сколько держать реплику на экране: примерно столько же её и произносят
-- (замер piper — около 0.055 с на символ), плюс вдох.
function SC.lineSeconds(text)
    return math.max(1.8, math.min(11.0, #tostring(text or '') * 0.055 + 1.2))
end

function SC.actorById(id)
    if id == nil or id == '' then return nil end
    local found = nil
    pcall(function()
        for _, a in ipairs(nearby.actors or {}) do
            local aid = ''
            pcall(function() aid = tostring(a.id or '') end)
            if aid == id then found = a; return end
        end
    end)
    return found
end

function SC.inCombat(obj)
    local fighting = false
    pcall(function()
        fighting = types.Actor.getStance(obj) == types.Actor.STANCE.Weapon
    end)
    return fighting
end

-- Состав: живые именованные NPC поблизости, которых игрок реально видит.
-- Квестовых помечаем — постановщик пустит их только в мирный повод.
function SC.collectCast()
    local cast, ppos = {}, self_.object.position
    pcall(function()
        for _, act in ipairs(nearby.actors or {}) do
            if #cast >= SC.CAST_MAX then return end
            if act ~= self_.object and act.type == types.NPC
               and not types.Actor.isDead(act)
               and (act.position - ppos):length() < SC.RADIUS
               and hasLineOfSight(act) then
                local ctx = buildNpcContext(act)
                if ctx.npc_id ~= '' and ctx.npc_name ~= '' then
                    -- Для СЦЕН защита нужна уже, чем для разговора. В разговоре
                    -- «есть канонные реплики» — разумный признак важности, но в
                    -- Сейда Нин такие есть у КАЖДОГО горожанина, и при широкой
                    -- мерке весь состав оказывался квестовым: ни одной драки,
                    -- ни одного вымогательства не случилось бы никогда.
                    -- Бережём тех, кто правда держит сюжет: с игровым скриптом
                    -- на себе или упомянутых в открытой записи журнала.
                    local story, why = isStoryCritical(act, ctx)
                    if why == 'канонные реплики' then story = false end
                    cast[#cast + 1] = {
                        id = ctx.npc_id, name = ctx.npc_name, race = ctx.npc_race,
                        class = ctx.npc_class, faction = ctx.npc_faction,
                        is_male = ctx.npc_is_male, story = story and true or false,
                    }
                end
            end
        end
    end)
    return cast
end

function SC.stop()
    if not SC.cur then return end
    for _, obj in pairs(SC.cur.actors or {}) do
        -- Вернуть к обычной жизни: сцена кончилась, а пакет «иди туда» остался
        -- бы. duration здесь в ИГРОВЫХ ЧАСАХ — ставим пару, чтобы пакет сам
        -- истёк, а не висел на персонаже до конца прохождения.
        pcall(function()
            obj:sendEvent('StartAIPackage', { type = 'Wander', distance = 256,
                                              duration = 2, cancelOther = true })
        end)
    end
    SC.cur = nil
end

-- kind — СПИСОК поводов, подходящих по обстановке (кто рядом, где, который
-- час). Какой из них случится и не окажется ли он фарсом, решает мост: там
-- живут ручки «опасность» и «нелепость», и их правки подхватываются на лету.
function SC.ask(kind, order)
    order = order or ''
    local now = core.getSimulationTime()
    if SC.cur then return false end
    if (now - SC.askedAt) < 20 then return false end   -- не сыпать запросами
    local cast = SC.collectCast()
    if #cast < (order ~= '' and 1 or 2) then
        if order ~= '' then showMsg('[сцена] Рядом слишком мало народу.') end
        return false
    end
    SC.askedAt = now
    reqCounter = reqCounter + 1
    local hour = math.floor((core.getGameTime() / 3600) % 24)
    local ok, enc = pcall(json.encode, {
        type = 'scene',
        req_id = 'scene-' .. SESSION_SALT .. '-' .. reqCounter,
        fit = kind or {}, order = order,
        cast = cast,
        location = (self_.object.cell and tostring(self_.object.cell.name or '')) or '',
        when = string.format('%02d:00', hour),
    })
    if not ok then return false end
    print('[MWAI_REQ] ' .. enc)
    return true
end

-- Такты пришли: запоминаем и начинаем играть.
function SC.apply(rec)
    if SC.cur then return end
    local beats = rec.scene
    if type(beats) ~= 'table' or #beats == 0 then return end
    local actors, live = {}, {}
    for _, b in ipairs(beats) do
        local obj = SC.actorById(tostring(b.id or ''))
        if obj then
            actors[tostring(b.id)] = obj
            live[#live + 1] = b
        end
    end
    if #live == 0 then return end     -- все разошлись, пока модель думала
    SC.cur = { beats = live, i = 0, phase = 'next', timer = 0, actors = actors,
               kind = tostring(rec.kind or '') }
    SC.lastAt = core.getSimulationTime()
    showMsg('[сцена] ' .. tostring(#live) .. ' реплик — смотри вокруг')
end

-- Один шаг сцены за кадр. Такт: дойти (если сказано) -> сказать -> сделать.
function SC.step(dt)
    local s = SC.cur
    if not s then return end
    -- Игрок вмешался или ушёл — сцена не должна идти у него за спиной.
    if isOpen or voiceTalking then SC.stop(); return end

    s.timer = s.timer + dt

    if s.phase == 'next' then
        s.i = s.i + 1
        local b = s.beats[s.i]
        if not b then SC.stop(); return end
        s.cur = b
        local me = s.actors[tostring(b.id or '')]
        if not me then return end                 -- пропал: следующий такт
        local dest = s.actors[tostring(b.walk_to or '')]
        if dest and not SC.inCombat(me) then
            local far = false
            pcall(function() far = (me.position - dest.position):length() > SC.ARRIVED end)
            if far then
                pcall(function()
                    me:sendEvent('StartAIPackage', { type = 'Travel',
                        destPosition = dest.position, cancelOther = true })
                end)
                s.phase, s.timer = 'walk', 0
                return
            end
        end
        s.phase, s.timer = 'say', 0
        return
    end

    if s.phase == 'walk' then
        local b = s.cur
        local me   = s.actors[tostring(b.id or '')]
        local dest = s.actors[tostring(b.walk_to or '')]
        local close = false
        pcall(function()
            close = me and dest and (me.position - dest.position):length() <= SC.ARRIVED
        end)
        -- Сторож: дорога может не сложиться (дверь, обрыв, застрял в мебели).
        -- Без него сцена молча висела бы до конца сессии.
        if close or s.timer > SC.WALK_MAX then s.phase, s.timer = 'say', 0 end
        return
    end

    if s.phase == 'say' then
        local b = s.cur
        local who = tostring(b.name or 'NPC')
        pushScene(who, tostring(b.line or ''), '')
        showMsg(who .. ': ' .. tostring(b.line or ''))
        local d = 0
        pcall(function()
            local me = s.actors[tostring(b.id or '')]
            d = (me.position - self_.object.position):length()
        end)
        -- Озвучка через мост: он один знает голоса и очередь реплик.
        reqCounter = reqCounter + 1
        local ok, enc = pcall(json.encode, {
            type = 'scene_say',
            req_id = 'say-' .. SESSION_SALT .. '-' .. reqCounter,
            npc_id = tostring(b.id or ''), npc_name = who,
            npc_is_male = b.is_male ~= false, npc_race = tostring(b.race or ''),
            text = tostring(b.line or ''), distance = math.floor(d),
        })
        if ok then print('[MWAI_REQ] ' .. enc) end
        s.hold = SC.lineSeconds(b.line)
        s.phase, s.timer = 'hold', 0
        return
    end

    if s.phase == 'hold' then
        if s.timer < (s.hold or 2.0) then return end
        local b = s.cur
        local act = tostring(b.action or 'none')
        local me = s.actors[tostring(b.id or '')]
        local victim = s.actors[tostring(b.target or '')]
        -- ВТОРОЙ РУБЕЖ, и он обязателен. Обработчик действий общий с разговором,
        -- а там всё считается ОТ ИГРОКА: у attack цель по умолчанию — игрок.
        -- Драка без настоящего противника в сцене посторонних обернулась бы
        -- нападением на него ни за что. Нет цели — нет действия.
        if act == 'attack' and not victim then
            act = 'none'
        end
        if act ~= 'none' and me then
            -- Исполняем ТЕМ ЖЕ путём, что и действия из разговора: один
            -- обработчик, одни защиты, никакой второй реализации.
            pcall(function()
                execAction(act, '', me, tostring(b.name or ''), victim, '', '')
            end)
        end
        s.phase, s.timer = 'next', 0
        return
    end
end

-- Случайные поводы. Условия смотрим сами: модель мира не видит.
function SC.classHas(cast, word)
    local n = 0
    for _, c in ipairs(cast) do
        if tostring(c.class or ''):lower():find(word, 1, true) then n = n + 1 end
    end
    return n
end

function SC.pickKind()
    local cast = SC.collectCast()
    if #cast < 2 then return nil end
    local hour = math.floor((core.getGameTime() / 3600) % 24)
    local cellName, interior = '', false
    pcall(function()
        local c = self_.object.cell
        cellName = tostring(c and c.name or ''):lower()
        interior = (c ~= nil) and (c.isExterior == false)
    end)
    local quest = 0
    for _, c in ipairs(cast) do if c.story then quest = quest + 1 end end

    local pool = {}
    if SC.classHas(cast, 'guard') >= 2 and hour >= 7 and hour <= 20 then
        pool[#pool + 1] = 'drill'
    end
    if SC.classHas(cast, 'trader') + SC.classHas(cast, 'pawn') >= 2 then
        pool[#pool + 1] = 'merchant_row'
    end
    if SC.classHas(cast, 'priest') + SC.classHas(cast, 'monk') >= 1 and #cast >= 3 then
        pool[#pool + 1] = 'sermon'
    end
    if interior and #cast >= 2 then
        pool[#pool + 1] = 'domestic'
        -- Драка — только в питейном заведении и только без квестовых.
        local tavern = cellName:find('таверн', 1, true) or cellName:find('трактир', 1, true)
                    or cellName:find('корчм', 1, true) or cellName:find('клуб', 1, true)
        if tavern and quest == 0 and hour >= 17 then pool[#pool + 1] = 'tavern_brawl' end
    end
    if (not interior) and (hour >= 21 or hour < 5) and #cast >= 3
       and SC.classHas(cast, 'guard') == 0 and quest == 0 then
        pool[#pool + 1] = 'shakedown'
    end
    if #cast >= 3 then pool[#pool + 1] = 'gossip_ring' end
    pool[#pool + 1] = 'stranger'
    -- Отдаём ВЕСЬ список подходящих по обстановке, а не один повод: какой из
    -- них случится, решает мост — там живёт ручка «опасность», и её правки
    -- подхватываются на лету.
    return pool
end

function SC.tryEvent()
    if SC.cur or isOpen or voiceTalking then return end
    local now = core.getSimulationTime()
    if (now - SC.lastAt) < SC.COOLDOWN * SC.rate() then return end
    if math.random() > SC.CHANCE then return end
    local fit = SC.pickKind()
    if not fit or #fit == 0 then return end
    if SC.ask(fit, '') then SC.lastAt = now end
end

-- ── Proactive hails: sometimes an NPC decides to address the player first ─────

local PROACTIVE_RADIUS       = 320
local PROACTIVE_GLOBAL_CD    = 720    -- пауза между любыми окликами
local PROACTIVE_PER_NPC_CD   = 3600   -- per-NPC repeat cooldown
local PROACTIVE_CHANCE       = 0.02   -- шанс на проверку раз в 3 с
local proactiveTimer         = 0
local lastProactiveAt        = -math.huge
local proactiveSeen          = {}     -- npc_id -> sim time of last hail

local function tryProactiveHail()
    if isOpen then return end
    local now = core.getSimulationTime()
    if (now - lastProactiveAt) < PROACTIVE_GLOBAL_CD * SC.rate() then return end
    if math.random() > PROACTIVE_CHANCE then return end
    local npc = findNearestNpc()
    if not npc then return end
    pcall(function()
        if (npc.position - self_.object.position):length() > PROACTIVE_RADIUS then npc = nil end
    end)
    if not npc or npc == companionObj then return end
    local ctx = buildNpcContext(npc)
    if ctx.npc_id == '' then return end
    local seen = proactiveSeen[ctx.npc_id] or -math.huge
    if (now - seen) < PROACTIVE_PER_NPC_CD then return end
    proactiveSeen[ctx.npc_id] = now
    lastProactiveAt = now
    -- Lock the NPC (so H opens the conversation with them) and ask the bridge
    -- for a self-initiated line: shown as subtitles + voice via applyReply.
    lockedCtx    = ctx
    lockedNpcObj = npc
    sendRequest({
        type = 'lock_npc',
        npc_id = ctx.npc_id, npc_name = ctx.npc_name, npc_race = ctx.npc_race,
        npc_class = ctx.npc_class, npc_faction = ctx.npc_faction, location = ctx.location,
        npc_is_male = ctx.npc_is_male,
    })
    lastReplyText = '(' .. (ctx.npc_name ~= '' and ctx.npc_name or 'кто-то') .. ' окликает тебя...)'
    lastSpeaker   = ''
    lastEmotion   = ''
    inputBuffer   = ''
    sendMessage('__proactive__')
    -- Окно НЕ открываем. Тебя окликнули на улице — это реплика в воздух, а не
    -- начало беседы: она приходит субтитрами внизу, и можно просто пройти
    -- мимо. Захочешь ответить — H рядом с ним, разговор уже начат.
end

-- onFrame fires every rendered frame, INCLUDING while the game is paused in the
-- chat window — so replies arrive live without leaving the dialogue.
local function onFrame(dt)
    dt = dt or 0
    proactiveTimer = proactiveTimer + dt
    if proactiveTimer >= 3.0 then proactiveTimer = 0; pcall(tryProactiveHail) end
    surrenderTimer = surrenderTimer + dt
    if surrenderTimer >= 2.0 then surrenderTimer = 0; pcall(trySurrender) end
    watchTimer = watchTimer + dt
    if watchTimer >= 5.0 then
        watchTimer = 0
        pcall(worldWatch)
        pcall(handleCompanionDeath)
        pcall(investigateDeeds)   -- тёмные дела всплывают спустя время
    end
    contractTimer = contractTimer + dt
    if contractTimer >= 1.0 then contractTimer = 0; pcall(contractWatch) end
    pcall(combatWatch)   -- every frame: catch real damage ticks
    radiantTimer = radiantTimer + dt
    -- Радиантная болтовня и повод для сцены ищутся в одном ритме и по одному
    -- условию: игрок должен это видеть, иначе запрос потрачен впустую.
    if radiantTimer >= 5.0 then
        radiantTimer = 0
        pcall(radiantScan)
        pcall(SC.tryEvent)
    end
    threatTimer = threatTimer + dt
    if threatTimer >= 1.5 then threatTimer = 0; pcall(threatWatch) end
    theftTimer = theftTimer + dt
    if theftTimer >= 2.5 then
        local step = theftTimer; theftTimer = 0
        pcall(function() theftWatch(step) end)
    end
    -- Ambient radiant lines: poll + drain one line at a time.
    speechPollTimer = speechPollTimer + dt
    if speechPollTimer >= 1.0 then speechPollTimer = 0; pollNpcSpeech() end
    if #speechQueue > 0 then
        speechTimer = speechTimer + dt
        if speechTimer >= 3.5 then
            speechTimer = 0
            showMsg(table.remove(speechQueue, 1))
        end
    end
    -- Dialogue replies: poll ALWAYS (voice mode, companion/bystander lines and
    -- post-action replies arrive with the window closed — audit finding 1.3).
    pollTimer = pollTimer + dt
    if pollTimer >= 0.2 then pollTimer = 0; pollReply(); pollVoiceCue() end
    if radiantTimer > 4.5 then SC.pollTuning() end
    -- Сцена идёт по своим тактам; повод для новой ищем в том же ритме, что и
    -- радиантную болтовню, и с тем же условием — игрок должен это видеть.
    SC.step(dt)
    -- Nobody answering? Say so instead of leaving the player staring at a
    -- silent NPC: the usual cause is the game started without the bridge
    -- (openmw.exe launched directly instead of through the shortcut).
    if waitTimer >= 0 then
        waitTimer = waitTimer + dt
        -- Порог с запасом на свою модель: она честно думает 10-20 секунд, и на
        -- прежних 25 предупреждение вылезало посреди нормального разговора,
        -- пугая поломкой там, где всё работало.
        if waitTimer > 60 and not waitWarned then
            waitWarned = true
            pushScene('[мод]', 'Мост не отвечает уже минуту. Запущен ли он? ' ..
                'Игру нужно запускать ярлыком «Morrowind AI», а не openmw.exe напрямую.', '')
            refreshWindow()
        end
    end
end

-- Middle mouse button: toggle a free cursor over the running game (no pause,
-- no chat window). Lets you click the chat field or just have a cursor.
local function onMouseButtonPress(button)
    if button ~= 2 then return end   -- 2 = middle button (wheel click)
    if isOpen then
        -- Toggle the typing cursor; the chat window itself stays either way.
        if cursorActive then
            cursorActive = false
            leaveCursorMode()        -- keys go back to character controls
        else
            cursorActive = true
            enterCursorMode()        -- cursor appears; click the field and type
        end
        return
    end
    if cursorOnly then
        cursorOnly = false
        leaveCursorMode()
    else
        cursorOnly = true
        enterCursorMode()
    end
end

-- Sent by the engine's ui.lua whenever the UI mode stack changes. If OUR
-- cursor mode was dropped externally (Esc while typing, another menu opened),
-- reset the flags and restore vanilla pause-on-interface behaviour.
local function onUiModeChanged(data)
    local newMode = data and data.newMode or nil
    if (cursorActive or cursorOnly) and newMode ~= INTERFACE_MODE then
        cursorActive = false
        cursorOnly = false
        pcall(function() I.UI.setPauseOnMode(INTERFACE_MODE, true) end)
    end
end

local function onInit()
    showMsg('[ИИ-мод] H — диалог с NPC; колёсико мыши — курсор.')
end

-- Persist per-NPC dialogue history INSIDE the savegame, so loading an earlier
-- save also rewinds what NPCs remember (no "memory from the future").
local function onSave()
    return {
        npcHistory = npcHistory, companionObj = companionObj, companionCtx = companionCtx,
        npcMood = npcMood, npcFacts = npcFacts, worldRumors = worldRumors,
        companionLoss = companionLoss, debts = debts,
        escort = escort, duel = duel, npcFate = npcFate, npcArc = npcArc,
        dirtyDeeds = dirtyDeeds,
    }
end

local function onLoad(data)
    npcHistory = (data and data.npcHistory) or {}
    companionObj = data and data.companionObj or nil
    companionCtx = data and data.companionCtx or nil
    npcMood = (data and data.npcMood) or {}
    npcFacts = (data and data.npcFacts) or {}
    worldRumors = (data and data.worldRumors) or {}
    companionLoss = data and data.companionLoss or nil
    debts = (data and data.debts) or {}
    escort = data and data.escort or nil
    duel = data and data.duel or nil
    npcFate = (data and data.npcFate) or {}
    npcArc = (data and data.npcArc) or {}
    dirtyDeeds = (data and data.dirtyDeeds) or {}
end

return {
    engineHandlers = {
        onInit             = onInit,
        onKeyPress         = onKeyPress,
        onKeyRelease       = onKeyRelease,
        onMouseButtonPress = onMouseButtonPress,
        onFrame            = onFrame,
        onQuestUpdate      = onQuestUpdate,
        onSave             = onSave,
        onLoad             = onLoad,
    },
    eventHandlers = {
        UiModeChanged = onUiModeChanged,
    },
}
