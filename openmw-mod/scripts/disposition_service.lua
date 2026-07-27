-- disposition_service.lua (GLOBAL)
-- Applies LLM-driven disposition deltas to the ENGINE's 0-100 scale.
-- setBaseDisposition is only callable from global scripts, so the player
-- script sends us an event. Once applied, all vanilla mechanics follow:
-- barter prices, service refusals, guard tolerance, voiced barks.

local types = require('openmw.types')
local world = require('openmw.world')
local util  = require('openmw.util')
local core  = require('openmw.core')
local I     = require('openmw.interfaces')

-- Audit 4.1: daily disposition budget per NPC — praise/flattery can move the
-- 0-100 scale only so far per game day (mirrors vanilla Speechcraft pacing).
local dispBudget = {}   -- npcId -> {day=..., up=..., down=...}
local DISP_DAILY_UP, DISP_DAILY_DOWN = 12, 20

-- NPCs departing for good: walk away, then get removed from the world.
local departures = {}   -- { {obj=..., t=<seconds left>}, ... }

local function onSetDisposition(data)
    if not data or not data.npc then return end
    local delta = tonumber(data.delta) or 0
    if delta == 0 then return end
    local player = world.players and world.players[1]
    if not player then return end
    pcall(function()
        -- Daily budget: cap how far talk alone can move this NPC per game day.
        local id = tostring(data.npc.id or data.npc.recordId or '?')
        local day = math.floor((core.getGameTime() or 0) / 86400)
        local b = dispBudget[id]
        if not b or b.day ~= day then b = { day = day, up = 0, down = 0 }; dispBudget[id] = b end
        if delta > 0 then
            delta = math.min(delta, DISP_DAILY_UP - b.up)
            if delta <= 0 then return end
            b.up = b.up + delta
        else
            delta = math.max(delta, -(DISP_DAILY_DOWN - b.down))
            if delta >= 0 then return end
            b.down = b.down - delta
        end
        local base = types.NPC.getBaseDisposition(data.npc, player)
        local new = math.max(0, math.min(100, base + delta))
        types.NPC.setBaseDisposition(data.npc, player, new)
        print(string.format('[morrowind-ai][disp] %s: %d -> %d (%+d)',
            tostring(data.npc.recordId or '?'), base, new, delta))
    end)
end

-- Hand REAL gold to the player (spoken promises must be kept). If the giving
-- NPC actually carries gold, it is honestly deducted from their purse first.
local function onGiveGold(data)
    local amount = math.floor(tonumber(data and data.amount) or 0)
    if amount < 1 then return end
    if amount > 500 then amount = 500 end
    local player = world.players and world.players[1]
    if not player then return end
    pcall(function()
        -- Audit 1.4: NEVER mint money beyond the NPC's real purse (allow only a
        -- small "pocket change" allowance of 25 the engine does not model).
        local purse = 0
        local stack = nil
        if data.npc and data.npc:isValid() then
            stack = types.Actor.inventory(data.npc):find('gold_001')
            purse = (stack and stack.count) or 0
        end
        local give = math.min(amount, purse + 25)
        if give < 1 then return end
        if stack and purse > 0 then
            stack:remove(math.min(purse, give))
        end
        local gold = world.createObject('gold_001', give)
        gold:moveInto(types.Actor.inventory(player))
        print('[morrowind-ai][gold] player received ' .. tostring(give) ..
              ' (asked ' .. tostring(amount) .. ', purse ' .. tostring(purse) .. ')')
    end)
end

-- Take gold FROM the player (they offered, the NPC accepted). The gold really
-- moves into the NPC's inventory — kill them later and it's still there.
local function onTakeGold(data)
    local amount = math.floor(tonumber(data and data.amount) or 0)
    if amount < 1 then return end
    if amount > 500 then amount = 500 end
    local player = world.players and world.players[1]
    if not player then return end
    pcall(function()
        local stack = types.Actor.inventory(player):find('gold_001')
        if not stack or (stack.count or 0) < amount then return end
        stack:remove(amount)
        if data.npc and data.npc:isValid() then
            world.createObject('gold_001', amount):moveInto(types.Actor.inventory(data.npc))
        end
        print('[morrowind-ai][gold] player paid ' .. tostring(amount))
    end)
end

-- Report the player's offense through the VANILLA crime system: bounty +
-- guards arrive with the normal arrest dialogue (pay fine / jail / resist).
-- Один донос за раз. Предохранитель, а не украшение: детектор кражи однажды
-- обвинил игрока семнадцать раз подряд (по разу на каждую вещь на полке), и
-- каждое обвинение вешало ОТДЕЛЬНОЕ нападение. Игрок сел с штрафом 200, вышел
-- и тут же сел снова с 40 — доносы всё ещё шли.
--
-- Сам залп исправлен там, где он рождался, но закон не должен зависеть от того,
-- что никто наверху не ошибётся ещё раз.
local lastCrimeAt = -1e9
local CRIME_GAP = 20   -- секунд симуляции между доносами

local function onReportCrime(data)
    local player = world.players and world.players[1]
    if not player then return end
    local now = core.getSimulationTime and core.getSimulationTime() or 0
    if now - lastCrimeAt < CRIME_GAP then
        print('[morrowind-ai][crime] донос отклонён: предыдущий был '
              .. string.format('%.1f', now - lastCrimeAt) .. ' с назад')
        return
    end
    lastCrimeAt = now
    pcall(function()
        I.Crimes.commitCrime(player, {
            type = types.Player.OFFENSE_TYPE.Assault,
            victim = data and data.victim or nil,
            victimAware = true,
        })
        print('[morrowind-ai][crime] assault reported')
    end)
end

-- Temple absolution: a priest clears the player's outstanding bounty (the
-- tithe itself is moved by the normal GOLD path before this fires).
local function onAbsolve(data)
    local player = world.players and world.players[1]
    if not player then return end
    pcall(function()
        local bounty = types.Player.getCrimeLevel(player) or 0
        if bounty <= 0 then return end
        types.Player.setCrimeLevel(player, 0)
        print('[morrowind-ai][absolve] bounty cleared: ' .. tostring(bounty))
    end)
end

-- NPC leaves forever: walk away from the player, then vanish from the world.
local function onDepart(data)
    local npc = data and data.npc
    if not npc then return end
    pcall(function()
        if not npc:isValid() then return end
        local player = world.players and world.players[1]
        local dir = util.vector3(1, 1, 0)
        if player then
            local d = npc.position - player.position
            if d:length() > 1 then dir = d:normalize() end
        end
        -- Точку по ПРОХОДИМОЙ земле подбирает скрипт игрока: карта проходимости
        -- (nearby.*) доступна только там. Прямая линия осталась запасным
        -- вариантом — именно она уводила людей в море, когда «прочь от игрока»
        -- означало в воду.
        local dest = data.dest or (npc.position + dir * 4000)
        npc:sendEvent('StartAIPackage', {
            type = 'Travel', destPosition = dest, cancelOther = true,
        })
        departures[#departures + 1] = { obj = npc, t = 75 }
    end)
end

-- ── Тёмные дела: то, о чём игрок может договориться с NPC ───────────────────
-- Всё ниже меняет мир по-настоящему: урон, инвентари, замки, поведение стражи.

-- Slow poison in someone's cup. The engine has no "poison" potion, but it has
-- a Damage Health spell with a duration; applied with the poisoner as caster
-- and renamed, it is a real, lethal, visible active effect.
local POISON_SPELL = 'grave curse: health'

local function onPoison(data)
    local victim = data and data.victim
    if not victim then return end
    local doses = math.max(1, math.min(6, math.floor(tonumber(data.doses) or 2)))
    pcall(function()
        if not victim:isValid() or types.Actor.isDead(victim) then return end
        for _ = 1, doses do
            types.Actor.activeSpells(victim):add({
                id = POISON_SPELL, effects = { 0 }, name = 'Отравление',
                caster = data.caster, stackable = true,
            })
        end
        print(string.format('[morrowind-ai][poison] %s отравлен(а) x%d',
            tostring(victim.recordId or '?'), doses))
    end)
end

-- Move a real item between two inventories: a pickpocket, a plant, a swindle.
-- `hint` picks WHAT: a name fragment, or the most valuable thing they carry.
local function onMoveItem(data)
    local from, to = data and data.from, data and data.to
    if not from or not to then return end
    local hint = tostring(data.hint or ''):lower()
    pcall(function()
        if not (from:isValid() and to:isValid()) then return end
        local best, bestScore = nil, -1
        for _, it in ipairs(types.Actor.inventory(from):getAll()) do
            local okItem = false
            local name, value, equipped = '', 0, false
            pcall(function()
                local rec = it.type.record(it)
                name = tostring(rec.name or it.recordId or ''):lower()
                value = tonumber(rec.value) or 0
                equipped = types.Actor.hasEquipped(from, it)
                okItem = not equipped and it.recordId ~= 'gold_001'
            end)
            if okItem then
                local score = -1
                if hint ~= '' and hint ~= 'none' and name:find(hint, 1, true) then
                    score = 100000 + value          -- назвали конкретную вещь
                elseif hint == '' or hint == 'none' then
                    score = value                    -- иначе самое ценное
                end
                if score > bestScore then best, bestScore = it, score end
            end
        end
        if not best then
            print('[morrowind-ai][move] нечего брать')
            return
        end
        local nm = tostring(best.recordId)
        best:moveInto(types.Actor.inventory(to))
        print('[morrowind-ai][move] ' .. nm .. ': ' ..
            tostring(from.recordId or 'игрок') .. ' -> ' .. tostring(to.recordId or 'игрок'))
    end)
end

-- Somebody talks the lock open: a real unlocked door or chest.
local function onUnlock(data)
    local obj = data and data.object
    if not obj then return end
    pcall(function()
        if not obj:isValid() then return end
        types.Lockable.unlock(obj)
        print('[morrowind-ai][unlock] ' .. tostring(obj.recordId or '?'))
    end)
end

-- A frame-up: the evidence is already planted, now the law is told. Guards in
-- earshot turn on the victim for real — the engine's own combat, not a message.
local function onFrameVictim(data)
    local victim = data and data.victim
    if not victim then return end
    pcall(function()
        if not victim:isValid() then return end
        local n = 0
        for _, guard in ipairs(world.activeActors) do
            local isGuard = false
            pcall(function()
                isGuard = guard ~= victim and guard.type == types.NPC
                    and not types.Actor.isDead(guard)
                    and tostring(types.NPC.record(guard).class or ''):lower():find('guard') ~= nil
                    and (guard.position - victim.position):length() < 3000
            end)
            if isGuard then
                guard:sendEvent('StartAIPackage',
                    { type = 'Combat', target = victim, cancelOther = true })
                n = n + 1
            end
        end
        print('[morrowind-ai][frame] стражников подняли на ' ..
            tostring(victim.recordId or '?') .. ': ' .. n)
    end)
end

-- An abduction: the victim is made to follow their captor, who then walks off.
local function onAbduct(data)
    local victim, captor = data and data.victim, data and data.captor
    if not (victim and captor) then return end
    pcall(function()
        if not (victim:isValid() and captor:isValid()) then return end
        victim:sendEvent('StartAIPackage',
            { type = 'Follow', target = captor, cancelOther = true })
        local dir = util.vector3(1, 1, 0)
        local player = world.players and world.players[1]
        if player then
            local d = captor.position - player.position
            if d:length() > 1 then dir = d:normalize() end
        end
        captor:sendEvent('StartAIPackage', {
            type = 'Travel', destPosition = captor.position + dir * 3500,
            cancelOther = true,
        })
        print('[morrowind-ai][abduct] ' .. tostring(victim.recordId or '?') ..
            ' уведён(а) ' .. tostring(captor.recordId or '?'))
    end)
end

-- "Иди туда" / "жди меня здесь": a real Travel or a stand-still Wander.
local function onGoTo(data)
    local actor = data and data.actor
    if not actor then return end
    pcall(function()
        if not actor:isValid() then return end
        if data.stay then
            actor:sendEvent('StartAIPackage',
                { type = 'Wander', distance = 0, duration = 3600, cancelOther = true })
            print('[morrowind-ai][goto] ' .. tostring(actor.recordId or '?') .. ' ждёт на месте')
            return
        end
        if data.destPosition then
            actor:sendEvent('StartAIPackage', {
                type = 'Travel', destPosition = data.destPosition, cancelOther = true,
            })
            print('[morrowind-ai][goto] ' .. tostring(actor.recordId or '?') .. ' идёт к цели')
        end
    end)
end

local function onUpdate(dt)
    for i = #departures, 1, -1 do
        local d = departures[i]
        d.t = d.t - dt
        if d.t <= 0 then
            pcall(function()
                if d.obj:isValid() and not types.Actor.isDead(d.obj) then
                    print('[morrowind-ai][depart] ' .. tostring(d.obj.recordId) .. ' left the world')
                    d.obj:remove()
                end
            end)
            table.remove(departures, i)
        end
    end
end

-- Hand a REAL item from the NPC's inventory to the player (matched by name).
local function onGiveItem(data)
    local npc, query = data and data.npc, tostring(data and data.query or ''):lower()
    if not npc or query == '' then return end
    local player = world.players and world.players[1]
    if not player then return end
    pcall(function()
        if not npc:isValid() then return end
        -- Audit 4.2: never strip equipped gear; cap the giveaway value.
        local equipped = {}
        pcall(function()
            for _, eq in pairs(types.Actor.getEquipment(npc) or {}) do
                if eq then equipped[eq.id] = true end
            end
        end)
        for _, it in ipairs(types.Actor.inventory(npc):getAll() or {}) do
            local nm, value = '', 0
            pcall(function()
                local rec = it.type and it.type.record and it.type.record(it)
                nm = rec and tostring(rec.name or ''):lower() or ''
                value = (rec and tonumber(rec.value)) or 0
            end)
            if nm ~= '' and (nm:find(query, 1, true) or query:find(nm, 1, true)) then
                if equipped[it.id] then
                    print('[morrowind-ai][item] refused (equipped): ' .. nm)
                elseif value > 400 then
                    print('[morrowind-ai][item] refused (too valuable ' .. value .. '): ' .. nm)
                else
                    it:moveInto(types.Actor.inventory(player))
                    print('[morrowind-ai][item] player received: ' .. nm)
                end
                return
            end
        end
        print('[morrowind-ai][item] no match in inventory for: ' .. query)
    end)
end

-- Town travel-point index, built lazily from all NPC travel destinations
-- (silt strider / boat routes give us safe positions in every major town).
local RU_TOWNS = {
    ['балмор'] = 'balmora', ['вивек'] = 'vivec', ['альд'] = 'ald-ruhn',
    ['садрит'] = 'sadrith mora', ['гнисис'] = 'gnisis', ['кальдер'] = 'caldera',
    ['пелагиад'] = 'pelagiad', ['сейда'] = 'seyda neen', ['молаг'] = 'molag mar',
    ['суран'] = 'suran', ['хла оад'] = 'hla oad', ['гнаар'] = 'gnaar mok',
    ['дагон'] = 'dagon fel', ['тель'] = 'tel', ['эбенгард'] = 'ebonheart',
    ['вос'] = 'vos', ['маар ган'] = 'maar gan', ['хуул'] = 'khuul',
}
local travelIndex = nil

local function buildTravelIndex()
    travelIndex = {}
    pcall(function()
        for _, rec in ipairs(types.NPC.records) do
            for _, dest in ipairs(rec.travelDestinations or {}) do
                local cid = tostring(dest.cellId or ''):lower()
                if cid ~= '' and travelIndex[cid] == nil then
                    travelIndex[cid] = { cellId = dest.cellId, position = dest.position }
                end
            end
        end
    end)
    local n = 0
    for _ in pairs(travelIndex) do n = n + 1 end
    print('[morrowind-ai] travel index: ' .. n .. ' destinations')
end

local function onRelocate(data)
    local npc = data and data.npc
    if not npc then return end
    if travelIndex == nil then buildTravelIndex() end
    local town = tostring(data.town or ''):lower()
    local eng = nil
    for ru, en in pairs(RU_TOWNS) do
        if town:find(ru, 1, true) then eng = en; break end
    end
    local hit = nil
    if eng then
        for cid, d in pairs(travelIndex) do
            if cid:find(eng, 1, true) then hit = d; break end
        end
    end
    if hit then
        pcall(function()
            npc:teleport(hit.cellId, hit.position)
            npc:sendEvent('StartAIPackage', { type = 'Wander', distance = 512, cancelOther = true })
            print('[morrowind-ai][relocate] ' .. tostring(npc.recordId) .. ' -> ' .. tostring(hit.cellId))
        end)
    else
        -- Unknown destination (audit 1.1): DO NOT delete the NPC — they walk
        -- off a bit and stay in the world; the move stays narrative only.
        pcall(function()
            npc:sendEvent('StartAIPackage', { type = 'Wander', distance = 1024, cancelOther = true })
        end)
        print('[morrowind-ai][relocate] unknown town — NPC kept in world')
    end
end

-- ── Судьба героя: чем обернулась пощада или обещание ────────────────────────
-- Мало перенести человека в другой город — важно, КЕМ он там стал. Ищем ему
-- настоящее место по обитателям: лавку по торговцу, таверну по трактирщику.
local FATE_ROLES = {
    worker   = { hosts = { 'trader', 'pawnbroker', 'smith', 'apothecary', 'bookseller' },
                 props = {}, msg = 'пристроился работать' },
    drunk    = { hosts = { 'publican' },
                 props = { 'potion_cyro_brandy_01' }, msg = 'спивается' },
    innkeep  = { hosts = { 'publican' }, props = {}, msg = 'работает при таверне' },
    beggar   = { hosts = {}, props = {}, msg = 'побирается' },
    guard    = { hosts = { 'guard' }, props = {}, msg = 'подался в стражу' },
    smuggler = { hosts = {}, props = {}, msg = 'связался с тёмными людьми' },
    -- Зацикленная и нелепые. Движку важно только, у кого под боком поставить
    -- человека; всё остальное — рассказ, который идёт из его судьбы.
    ticket   = { hosts = {}, props = {}, msg = 'снова копит на билет' },
    actor    = { hosts = { 'publican' }, props = {}, msg = 'подался в комедианты' },
    prophet  = { hosts = {}, props = {}, msg = 'проповедует свой вещий сон' },
    fisher   = { hosts = {}, props = {}, msg = 'сидит на мостках с удочкой' },
    clerk    = { hosts = { 'bookseller', 'trader' }, props = {},
                 msg = 'устроился переписчиком' },
    guard_ic = { hosts = { 'guard' }, props = {}, msg = 'охраняет пустое место' },
}

local function findVenue(townEn, hosts)
    if #hosts == 0 then return nil end
    local found = nil
    pcall(function()
        for _, cell in ipairs(world.cells) do
            local nm = tostring(cell.name or ''):lower()
            if not cell.isExterior and nm ~= '' and nm:find(townEn, 1, true) then
                for _, npc in ipairs(cell:getAll(types.NPC)) do
                    local cls = ''
                    pcall(function() cls = tostring(types.NPC.record(npc).class or ''):lower() end)
                    for _, want in ipairs(hosts) do
                        if cls:find(want, 1, true) then
                            found = { cell = cell, position = npc.position }
                            return
                        end
                    end
                end
            end
        end
    end)
    return found
end

local function onSettleFate(data)
    local npc = data and data.npc
    if not npc then return end
    if travelIndex == nil then buildTravelIndex() end
    local role = FATE_ROLES[tostring(data.role or '')] or FATE_ROLES.beggar
    local town = tostring(data.town or ''):lower()
    local eng = nil
    for ru, en in pairs(RU_TOWNS) do
        if town:find(ru, 1, true) then eng = en; break end
    end
    if not eng then
        print('[morrowind-ai][fate] город не опознан: ' .. town)
        return
    end

    local venue = findVenue(eng, role.hosts)
    local dest = nil
    if venue then
        dest = { cellId = venue.cell.name, position = venue.position }
    else
        for cid, d in pairs(travelIndex) do          -- иначе на площадь города
            if cid:find(eng, 1, true) then dest = d; break end
        end
    end
    if not dest then
        print('[morrowind-ai][fate] некуда селить: ' .. eng)
        return
    end

    pcall(function()
        npc:teleport(dest.cellId, dest.position)
        npc:sendEvent('StartAIPackage',
            { type = 'Wander', distance = 250, duration = 100000, cancelOther = true })
        for _, recId in ipairs(role.props) do        -- реквизит новой жизни
            local it = world.createObject(recId, 2)
            if it then it:moveInto(types.Actor.inventory(npc)) end
        end
        print(string.format('[morrowind-ai][fate] %s -> %s (%s, %s)',
            tostring(npc.recordId or '?'), tostring(dest.cellId),
            tostring(data.role), role.msg))
    end)
end

return {
    engineHandlers = { onUpdate = onUpdate },
    eventHandlers = {
        MorrowindAiSetDisposition = onSetDisposition,
        MorrowindAiGiveGold       = onGiveGold,
        MorrowindAiTakeGold       = onTakeGold,
        MorrowindAiDepart         = onDepart,
        MorrowindAiGiveItem       = onGiveItem,
        MorrowindAiRelocate       = onRelocate,
        MorrowindAiReportCrime    = onReportCrime,
        MorrowindAiAbsolve        = onAbsolve,
        -- Тёмные дела
        MorrowindAiPoison         = onPoison,
        MorrowindAiMoveItem       = onMoveItem,
        MorrowindAiUnlock         = onUnlock,
        MorrowindAiFrame          = onFrameVictim,
        MorrowindAiAbduct         = onAbduct,
        MorrowindAiGoTo           = onGoTo,
        MorrowindAiSettleFate     = onSettleFate,
    },
}
