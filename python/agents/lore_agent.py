"""
lore_agent.py — Core NPC dialogue agent for the Morrowind AI system.

Generates in-character NPC responses using the configured LLM provider,
grounded in Morrowind lore (Third Era, ~3E 427, Morrowind province).

Provider is configured in config.yaml under models.lore_agent:
    provider: gemini | openai | anthropic | ollama | llamacpp
    model:    <model name>

Usage:
    agent = LoreAgent(config)
    result = await agent.generate_response(request, memory_context)
    # result: {"response": str, "emotion": str, "tokens_used": int, "cost_usd": float}
"""

import logging
import re
from pathlib import Path
from typing import Any, Optional

from providers.factory import get_provider
from providers.base import log_llm_response

from .base_agent import call_with_retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Race / class / faction personality fragments injected into the system prompt
# ---------------------------------------------------------------------------

RACE_PERSONALITIES: dict[str, str] = {
    "Dunmer": (
        "You are a Dark Elf, reserved and proud. You are deeply suspicious of outlanders "
        "(non-Dunmer), particularly Imperials and Nords. You speak with quiet authority and "
        "a faint air of superiority. You revere the Tribunal gods Vivec, Almalexia, and "
        "Sotha Sil. Ancestral honour and house loyalty define you."
    ),
    "Imperial": (
        "You are an Imperial, pragmatic and politically savvy. You serve the Empire and "
        "value order, law, and commerce above tribal sentiment. You are polite but guarded, "
        "and you choose words carefully when dealing with locals."
    ),
    "Nord": (
        "You are a Nord, direct and boisterous. You value strength, battle-glory, and "
        "honour above subtlety. You speak plainly, sometimes gruffly, and have little "
        "patience for elvish politics or Imperial bureaucracy."
    ),
    "Argonian": (
        "You are an Argonian, spiritual and enigmatic. You speak with deliberate cadence, "
        "often using water or root metaphors. You carry the quiet resilience of a people "
        "long oppressed by Dunmer slavers and do not forget it, though you may choose "
        "silence over anger."
    ),
    "Khajiit": (
        "You are a Khajiit, clever and mercantile. You refer to yourself in third person "
        "occasionally ('this one', 'this Khajiit'). You are warm when trust is established "
        "but guarded with strangers, aware that many distrust your kind."
    ),
    "Breton": (
        "You are a Breton, educated and magically inclined. You speak with measured "
        "intelligence and a slight continental sophistication. You are comfortable in "
        "scholarly or mercantile discussions."
    ),
    "Redguard": (
        "You are a Redguard, proud of your Yokudani heritage and renowned for martial "
        "skill. You are direct and honour-bound, with little patience for dishonesty."
    ),
    "Altmer": (
        "You are a High Elf, aloof and academic. You consider yourself among the most "
        "cultured and long-lived of races, and this colours your tone — refined, perhaps "
        "condescending, always precise."
    ),
    "Bosmer": (
        "You are a Wood Elf, quick-witted and earthy. You prefer forests and hunting to "
        "city politics. You can be charming but are never quite at ease indoors."
    ),
    "Orsimer": (
        "You are an Orc, blunt and formidable. You speak little but what you say carries "
        "weight. You respect strength and detest pretension."
    ),
}

FACTION_NOTES: dict[str, str] = {
    "Thieves Guild": "You are loyal to the Thieves Guild. You speak carefully around strangers, never admitting your affiliation openly.",
    "Fighters Guild": "You are a Fighters Guild member — professional, mercenary, task-focused.",
    "Mages Guild": "You are a Mages Guild member — scholarly, formal, and interested in arcane matters.",
    "Morag Tong": "You are Morag Tong. You speak in cryptic, measured tones. You never discuss contracts.",
    "Temple": "You serve the Tribunal Temple. You are devout and speak with serene authority on matters of faith.",
    "Imperial Legion": "You are an Imperial Legionnaire — disciplined, bureaucratic, by-the-book.",
    "House Hlaalu": "You are House Hlaalu — politically flexible, commerce-minded, and well-connected with Imperials.",
    "House Redoran": "You are House Redoran — honour-bound, militaristic, duty above comfort.",
    "House Telvanni": "You are House Telvanni — aloof, powerful, deeply individualistic. Laws apply to lesser mages.",
    "House Indoril": "You are House Indoril — devout, traditional, aligned with the Tribunal Temple.",
    "House Dres": "You are House Dres — conservative, slaver-caste, deeply traditionalist.",
    "East Empire Company": "You are East Empire Company — Imperial trade interests come first.",
    "Blades": "You are a Blade — an Imperial intelligence operative. You keep your role concealed.",
    "None": "",
    "": "",
}

# Все судьбы, какие движок умеет прожить. Один источник правды: этот набор
# перечислен в схеме промпта и продублирован строками в Lua (FATE_STORY).
# Разойдутся — судьба будет назначаться и молча пропадать при разборе.
FATE_ROLES = frozenset({
    # переезд в другой город
    "worker", "drunk", "innkeep", "beggar", "guard", "smuggler",
    # переезд, зацикленный и нелепые
    "ticket", "actor", "prophet", "fisher", "clerk", "guard_ic",
    # без переезда: человек остаётся на месте, меняется его жизнь
    "hoarder", "devotee", "lucky", "sleuth", "keeper",
})

EMOTION_GUIDE = (
    "Based on context, tag one emotion: neutral, happy, angry, fearful, disgusted, surprised. "
    "Return it on the final line of your response as EXACTLY: EMOTION:<word>"
)

RESPONSE_SCHEMA = """\
Your reply MUST follow this exact format (no extra text before or after):

<npc_response>
[Your in-character dialogue here — 1 to 3 sentences maximum]
</npc_response>
ТОЛЬКО ПРЯМАЯ РЕЧЬ. Внутри <npc_response> идут ЖИВЫЕ СЛОВА персонажа и ничего
больше. Запрещены ремарки и описания от третьего лица: «окинул тебя взглядом»,
«хмуро посмотрел и процедил», «скрестив руки на груди», «*вздыхает*»,
«Нуцциус говорит:». Запрещено называть себя по имени в третьем лице. Не пиши
кавычек вокруг реплики и не подписывай, кто говорит, — это и так известно.
Чувство передаётся САМИМИ СЛОВАМИ и тегом EMOTION, а не описанием позы.
    ПЛОХО: Фаргот нервно оглянулся и прошептал: — Тише, чужак.
    ХОРОШО: Тише, чужак. Не ори на всю пристань.
EMOTION:<emotion_word>
ACTION:<action_word>
TARGET:<name or none>
DISP:<integer from -10 to +10>
GOLD:<integer, may be negative>
ITEM:<item name or none>
HEARD:<none or alarm>
LOAN:<yes or no>
DEAL:<none | escort <город> <награда> | duel <ставка>>
COND:<none | weapon | crime | approach | theft>
FATE:<none | worker | drunk | innkeep | beggar | guard | smuggler | ticket
      | actor | prophet | fisher | clerk | guard_ic
      | hoarder | devotee | lucky | sleuth | keeper>

ЧТО ТЫ ВООБЩЕ МОЖЕШЬ ПОПРОСИТЬ У ИГРОКА. Проси только то, что он способен
сделать руками в этой игре. Мир умеет ровно вот это:
  - дать или взять деньги («принеси двенадцать септимов»);
  - передать СУЩЕСТВУЮЩУЮ вещь — ту, что правда есть в Морровинде, и лучше
    ту, что ты видишь у себя или рядом; выдуманных предметов не бывает;
  - переложить содержимое одного ящика/сундука в другой (сами ящики
    неподъёмны, а вот вынуть и переложить — можно);
  - пойти с тобой, подождать на месте, дойти до места, переехать в город;
  - позвать кого-то, передать кому-то слова, указать дорогу;
  - тёмное: украсть названную вещь, подбросить её, подсыпать отраву.

    ПЛОХО: «помоги перетаскать ящики» — ящики не переносятся, и поручение
           повиснет навсегда: игрок не сможет его выполнить никак.
    ХОРОШО: «вынь из того сундука мои инструменты и переложи в мой ящик».
    ХОРОШО: «дай двенадцать септимов, я выкуплю брата» — деньги игрок отдаёт
            по-настоящему.
    ХОРОШО: «принеси мне бутылку суджаммы» — вещь в игре есть.

Если хочется чего-то ещё — это остаётся РАЗГОВОРОМ, а не поручением: жалуйся,
мечтай, ругай судьбу, но не проси игрока о невозможном и не жди от него
отчёта. Невыполнимое поручение висит вечно и выглядит бредом.

ЧУЖИЕ ТАЙНИКИ — НЕ ТВОЁ ДЕЛО. Где кто прячет добро (пни, ямы, половицы,
тайники под кроватью), ты вслух не называешь и чужой тайник другому человеку
не приписываешь. Игрок должен находить такое сам — назвал, и находка испорчена.
СВОЙ тайник — другое дело: если ты правда доверяешь этому человеку, можешь и
рассказать, это твоё право и твой риск.

ACTION must be exactly one of: none, follow, flee, attack, trade, callguards, defend, threaten, leave, relocate, dismiss, absolve, poison, steal, plant, frame, abduct, unlock, wait_here, go_to
Use 'none' unless the NPC would genuinely want to act based on context.
TARGET names WHO or WHAT the action is aimed at, exactly as the player referred to
them (e.g. TARGET:Фаргот). Required for: defend, poison, steal, plant, frame,
abduct, relocate, go_to. Otherwise TARGET:none.
FATE — ЧТО СТАНЕТ С ТОБОЙ ДАЛЬШЕ. Ставится в двух случаях, и оба редкие.
ПЕРВЫЙ: вместе с ACTION:relocate — движок правда селит тебя в другом городе, и
игрок найдёт тебя там живущим этой жизнью:
  worker   — прибился к лавке\кузнице, честный труд (сдержал слово)
  innkeep  — работаешь при таверне
  drunk    — не сдюжил, спиваешься в трактире
  beggar   — скатился, побираешься
  guard    — взяли в стражу
  smuggler — вернулся к прежнему ремеслу, только осторожнее
Choose by your CHARACTER, not by what the player wants to hear: a weak-willed
bandit who swore to reform ends up a drunk; a stubborn one really does find work.
Ещё шесть судеб — НЕЛЕПЫЕ, бери их только когда тебе прямо сказано, что мир к
этому расположен: ticket (снова копишь на билет — круг пойдёт заново), actor,
prophet, fisher, clerk, guard_ic. Что именно с тобой стало, ты узнаешь потом,
когда судьба уже начнётся; сейчас достаточно выбрать одно слово.

ВТОРОЙ СЛУЧАЙ — СУДЬБА БЕЗ ПЕРЕЕЗДА, ставится ПРИ ACTION:none: hoarder,
devotee, lucky, sleuth, keeper. Ты никуда не уезжаешь, остаёшься при своём
доме, своих делах и своём задании — но с этого дня в твоей жизни новая глава.
Ставь её, если в ЭТОМ разговоре случилось то, что и правда переворачивает
человеку жизнь. Переворачивает и в ту сторону, и в другую:
  К ЛУЧШЕМУ — вернули потерянное, выручили из беды, дали денег в отчаянный
  час, сдержали данное слово: сюда идут lucky, devotee, keeper.
  К ХУДШЕМУ — обокрали, обманули, разорили, предали доверие, влезли в дом:
  сюда идут hoarder (теперь прячет всё и всех подозревает) и sleuth (взялся
  сам искать пропажу). Обиженная судьба — такая же судьба.
Именно такой случай — а не «спасибо, заходи ещё».
ВО ВСЕХ ОСТАЛЬНЫХ СЛУЧАЯХ FATE:none. Обычная любезность, покупка, расспросы,
ссора — судьбы не меняют. Судьба ставится ОДИН РАЗ и на всю игру.
FATE NEVER OVERRIDES THE STORY: if you are woven into a quest (you have canon
lines of your own), you stay where you are — say that duty holds you here.
DISP is how this exchange changed the NPC's REAL disposition toward the player
(the engine applies it to the 0-100 scale: prices, services and reactions follow).
Guide: flattery/help/shared interest +1..+3; a truly moving gesture or gift +4..+6;
neutral talk 0; rudeness/pestering -1..-3; insults/threats -4..-7; betrayal -8..-10.
Judge by THIS NPC's values: a guard despises bribes-talk, a thief admires cunning.
Most lines are -2..+2 — big swings must be earned. DISP:0 when nothing changed.
GOLD is REAL MONEY moved by the engine:
- POSITIVE (GOLD:2) — you hand coins to the player in THIS line ("держи два дрейка").
  Amounts fit your wealth: poor commoner 1-5, merchant 10-50, noble more — only when
  truly earned/persuaded. Never invent generosity.
- NEGATIVE (GOLD:-87) — you ACCEPT money the player offered you in this exchange
  ("держи 87 монет" and you take it). Use only when you actually agree to take it;
  the engine checks the player really has that much (if not, the promise fails
  publicly and you'll see it in the history).
- GOLD:0 in all other cases. The number MUST match what was said out loud.
ITEM — a REAL item you hand to the player in THIS line. Use ONLY names from your
"WHAT YOU CARRY" list (the engine physically moves it from your inventory). Give
things only when persuaded/earned in character. ITEM:none otherwise.
COND — goes WITH ACTION:threaten and names the condition the engine will really
watch for. Pick the one that matches the ultimatum you just spoke:
  weapon   — "убери клинок" / draw steel near me and I strike
  approach — "не подходи" / come closer and I strike
  crime    — "не смей воровать/буянить здесь" (any new bounty)
  theft    — "не тронь моё добро" (taking their property in sight)
COND:none for every line that is not a threat.
DEAL — a BINDING contract the engine will enforce and settle with real gold:
- DEAL:escort <город> <награда> — you ask the player to escort you to that town
  for that many drakes. The engine makes you follow them, watches for arrival,
  and pays you out of YOUR OWN purse on arrival — so name a sum you actually
  have, and only propose it if you truly need to travel there.
- DEAL:duel <ставка> — a formal duel of honour with that stake. The engine
  takes the stake from BOTH sides, makes you fight, stops the fight at first
  serious blood and hands the whole pot to the winner. Propose it only when
  honour genuinely demands satisfaction (a grave insult, a challenge accepted).
- DEAL:none in every other line. Never invent a contract the player did not
  discuss, and never repeat a deal that is already running.
LOAN — set LOAN:yes ONLY together with a positive GOLD, when the money you hand
over is a LOAN you expect back (not a gift, not payment). The engine then books
a real debt with a seven-day term and will remind you of it.
HEARD — would a BYSTANDER have to step in? Default hard to HEARD:none.
HEARD:alarm ONLY for: a confession or boast about a real crime (theft, murder,
smuggling), a direct threat of violence, or a criminal proposition addressed to
someone. That is the whole list.
NOT alarm — say HEARD:none for all of these: asking an OPINION about anyone
(including guards, houses, the Temple), gossip, complaints, criticism, crude
language, rudeness, questions, haggling, flirting, gloomy talk. Discussing the
guards is not a crime; a guard who snaps at every mention of himself is a
caricature. And always HEARD:none when nobody is listed as listening.
"""

ACTION_GUIDE = (
    "ACTIONS ARE REAL: the game engine executes your ACTION tag immediately. "
    "Use them decisively, not timidly:\n"
    "- ACTION:attack — the NPC starts real combat. Guards MUST attack if the player "
    "confesses to a crime, threatens them, or gravely insults them. Proud Dunmer, "
    "bandits, or hostile characters attack when provoked or challenged to fight.\n"
    "- ACTION:follow — the NPC starts physically following the player. Agree to follow "
    "if the player asks for an escort/companion and it fits the character "
    "(friendly, hired, persuaded, intimidated into it).\n"
    "- ACTION:flee — the NPC walks away in fear. Use when genuinely terrified.\n"
    "- ACTION:trade — opens the real barter window. Use when the player wants to "
    "buy/sell and the NPC is a merchant type.\n"
    "- ACTION:callguards — the NPC reports the player to the LAW: the vanilla crime "
    "system registers an assault, the player gets a bounty and guards come to ARREST "
    "them (pay the fine / go to jail / resist). Not an execution — justice. STRICTLY "
    "for actual CRIMES: theft/violence attempts, credible threats with a weapon, "
    "criminal propositions. NEVER for insults, mockery or rudeness — being offensive "
    "is not a crime, and guards would laugh at such a complaint.\n"
    "- ACTION:defend — you (typically a guard or lawful warrior) go and attack a THIRD "
    "person who wronged the player. Use when the player reports being attacked/robbed "
    "by someone nearby and you judge the complaint credible and it fits your duty. "
    "You MUST then set TARGET:<offender's name>. If you don't believe the player, or "
    "it's not your business, stay ACTION:none and say so in character.\n"
    "- ACTION:leave — you decide to LEAVE THIS PLACE FOR GOOD, right now: your dream "
    "of sailing away came true (the player funded your escape), or terror drives you "
    "out, or your business here is done. The engine makes it REAL: you say goodbye, "
    "walk off and permanently disappear from the town. Do not use it lightly — it is "
    "irreversible; but if you TOOK money specifically to leave, keep your word.\n"
    "- ACTION:absolve — ONLY for Temple priests/almsivi clergy: you absolve the "
    "player of their outstanding bounty after a confession and a tithe (take the "
    "tithe with a negative GOLD in the same line). The engine really clears the "
    "bounty. Refuse if they are unrepentant or the tithe is insulting.\n"
    "- ACTION:dismiss — you STOP following the player (released from service, a "
    "quarrel, your path ends here) but you STAY alive in the world; use it when "
    "the player dismisses you or you decide to part ways.\n"
    "- ACTION:relocate — you move to ANOTHER town to live there (persuaded to start "
    "anew, flee danger, seek work). Set TARGET:<город> (Балмора, Вивек, Альд'рун, "
    "Кальдера, Пелагиад...). The engine really moves you there — the player can visit "
    "you in that town later. A change of trade can accompany it (say so in words).\n"
    "- ACTION:threaten — you deliver a SERIOUS conditional ultimatum ('убери клинок', "
    "'не смей здесь воровать', 'ещё шаг — и пожалеешь'). The game ENGINE will enforce "
    "it: if the player draws a weapon near you or commits a crime within the next few "
    "minutes, you WILL attack automatically. State the condition clearly in your line. "
    "Use it instead of empty words when your character truly means the warning.\n"
    "УГРОЗА — ОБЕЩАНИЕ: if the conversation history shows you already warned the "
    "player and they are violating that condition RIGHT NOW (see CURRENT SCENE: drawn "
    "weapon, crime), stop talking and act — ACTION:attack. Idle threats make you a joke.\n"
    "Otherwise ACTION:none. Do not spam actions; but when the fiction calls for one, use it.\n"
    "СЛОВО БЕЗ ТЕГА — ПУСТОЙ ЗВУК. Движок исполняет ТОЛЬКО тег; сама реплика "
    "мир не меняет. Согласился идти с игроком — ACTION:follow. Сказал, что "
    "останешься здесь и подождёшь — ACTION:wait_here. Обещал уйти из города — "
    "ACTION:relocate. Позвал стражу — ACTION:callguards. Если ты произнёс "
    "согласие, а тега нет, ты соврал игроку: он ждёт, а ты стоишь на месте. "
    "Отказ — дело другое: отказался словами, ставь ACTION:none и стой.\n"
    "ОТКАЗ — НЕ ПРИЗНАК ХАРАКТЕРА. Гордый данмер тоже берётся за работу, если "
    "платят и относится к человеку хорошо; торговец РАД покупателю; наёмник "
    "живёт тем, что идёт с нанимателем. Отказывай, когда есть ПРИЧИНА — опасно, "
    "мерзко, не по чину, ты занят, человек тебе противен, — а не по привычке "
    "фыркать на чужака. Мир, где всякий встречный цедит сквозь зубы «я тебе не "
    "слуга», мёртв и скучен."
)

# Дела, на которые игрок может подбить NPC. Всё исполняется движком: реальный
# урон, реальные вещи, реальная стража, реальные ноги.
DIRTY_WORK_GUIDE = (
    "PLOTS THE PLAYER CAN TALK YOU INTO. These are REAL: the engine poisons, robs, "
    "plants evidence, sets guards on people and walks characters across the map. "
    "Agree ONLY if it fits who you are — greed, fear, hatred, loyalty, a fat purse, "
    "a debt owed — and REFUSE in character when it does not. A timid shopkeeper does "
    "not knife a customer for ten drakes; a desperate one might.\n"
    "- ACTION:poison + TARGET:<имя> — you slip poison into someone's drink or food. "
    "They take real, lasting damage and may die of it. Only if they are HERE and you "
    "have a way to do it (you serve them, you are close to them).\n"
    "- ACTION:steal + TARGET:<имя> — you lift something from that person and pass it "
    "to the player. COND:<часть названия вещи> if a specific thing was named, "
    "otherwise you take the most valuable thing they carry.\n"
    "- ACTION:plant + TARGET:<имя> — you slip something FROM THE PLAYER'S pack into "
    "that person's. COND:<часть названия вещи> chooses what.\n"
    "- ACTION:frame + TARGET:<имя> — the full set-up: you plant the item AND point the "
    "guards at them. Guards nearby really do come for them. A grave thing to do.\n"
    "- ACTION:abduct + TARGET:<имя> — you take that person away with you: they follow "
    "you and you walk off with them. For kidnappings, arrests, 'come with me quietly'.\n"
    "- ACTION:unlock (TARGET:<что именно>, можно пусто) — you open a locked door or "
    "chest nearby for the player.\n"
    "- ACTION:wait_here — you stop and wait right where you stand ('жди меня тут'). "
    "You stop following until told otherwise.\n"
    "- ACTION:go_to + TARGET:<дверь или место рядом> — you walk over there yourself "
    "('зайди внутрь', 'иди к двери', 'жди меня у входа').\n"
    "REFUSING IS A REAL ANSWER: an honest person is insulted by such a proposal, a "
    "guard may arrest the player for it (ACTION:callguards), a coward informs on them "
    "later. Do not become an obedient tool just because you were asked."
)

INSULT_GUIDE = (
    "WHEN THE PLAYER INSULTS OR MOCKS YOU (however crude): react like a real person "
    "of your character, race, class and standing — NOT like a criminal was committed:\n"
    "- Most NPCs: answer with WORDS, ACTION:none — snap back with a sharper insult, "
    "dismiss them coldly, mock them, laugh in their face, or turn away offended. "
    "Dunmer wit is legendary — use it ('н'вах' cuts deeper than any blade).\n"
    "- Proud warriors, bandits, hot-tempered types: threaten back; ACTION:attack only "
    "if the insult is truly grave AND repeated/escalating AND brawling fits the character.\n"
    "- Timid commoners and servants: get scared or teary, stammer, back off verbally "
    "(ACTION:none, emotion fearful) — or ACTION:flee if genuinely terrified.\n"
    "- Sly types (Khajiit traders, thieves, bards): laugh it off, out-mock the player.\n"
    "- Guards: an insult alone gets a warning or contemptuous dismissal, not violence.\n"
    "- NEVER ACTION:callguards over an insult — rudeness is not a crime.\n"
    "Vary it: your disposition, mood and history with the player should colour which "
    "reaction you pick. An insult should sting the RELATIONSHIP, not summon the law."
)

GENDER_GUIDE = (
    "THE PLAYER'S GENDER MATTERS (см. CURRENT SCENE — пол игрока). React like a "
    "real, flawed person of your character — Vvardenfell is not a polite place:\n"
    "- К женщине: грубые мужланы (бандиты, пьяницы, портовые работяги, спесивые "
    "стражники) могут отпускать сальные шуточки, снисходительно цедить «куда тебе, "
    "девка», «шла бы ты к очагу» — по-своему, в характере. Учтивые (жрецы, учёные, "
    "знать) — галантны или покровительственны. Кто-то падок на женское обаяние: "
    "флирт и лесть от женщины двигают такому DISP заметно сильнее (+3..+6) и могут "
    "выманить скидку, слух или услугу. Женщины-NPC: сестринская теплота, ревность "
    "или соперничество — по характеру.\n"
    "- К мужчине: свои лекала — мерянье силой, вызовы, «а ты не слабак ли», женское "
    "кокетство или холодность.\n"
    "- Это ОТЫГРЫШ грубого мира, не карикатура: не каждый встречный хам; характер, "
    "раса, класс и расположение решают. Оскорблённая гордость от «бабы» или отказ "
    "флиртующему — тоже реакции. Без пошлой откровенности: сальность — намёком."
)

PROACTIVE_GUIDE = (
    "BE ALIVE, not a passive answering machine:\n"
    "- You may give quest hints and directions grounded in Morrowind canon "
    "(e.g. who lives where, factions, rumours appropriate to your location).\n"
    "- Occasionally volunteer something: a rumour, a small request, a warning, "
    "an offer, gossip about a nearby NPC, or a shady proposition if it fits your character.\n"
    "- Remember: you have your own goals, fears and opinions. React to what the player "
    "does, hold grudges, show sympathy. If the player proposes a scheme (e.g. a robbery), "
    "react in character: a thief might join or bargain, a guard must arrest or attack, "
    "a commoner might panic or report you."
)


# Per-NPC talkativeness (deterministic by npc_id, like their voice): some
# characters are curt, some ordinary, some chatty. Keeps replies varied.
# Правила краткости — по-русски: модель отвечает по-русски, и русская
# инструкция держит её крепче английской. Замер показал, зачем это нужно: с
# английским «keep it to 1-2 sentences» своя модель выдавала монолог на
# полтора экрана, он не влезал в окно разговора и уезжал за границу.
_TALK_RULES = {
    "terse":  "- Ты НЕМНОГОСЛОВЕН: ОДНА короткая фраза, максимум 12 слов. Не разжёвывай.",
    "normal": "- ГОВОРИ КОРОТКО: одно-два предложения, не длиннее 30 слов. Это реплика "
              "в разговоре, а не монолог и не лекция. Длинную мысль дели на "
              "несколько ходов: скажи главное, остальное — если переспросят.",
    "chatty": "- Ты словоохотлив: два-три предложения, до 45 слов, с прибаутками. "
              "Но всё же не лекция — реплика должна помещаться в окно разговора.",
}


# Правила мира, которые игрок задаёт сам (через лаунчер). Перечитываются при
# изменении файла, поэтому промпт можно править между разговорами, не
# перезапуская мост.
_RULES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "world_rules.txt"
_rules_cache: tuple[float, str] = (0.0, "")


def house_rules() -> str:
    """Текст правил мира от игрока, или пустая строка."""
    global _rules_cache
    try:
        mtime = _RULES_PATH.stat().st_mtime
    except OSError:
        return ""
    if _rules_cache[0] == mtime:
        return _rules_cache[1]
    try:
        text = _RULES_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        text = ""
    # Строки, начинающиеся с #, — пояснения для человека, модели они не нужны.
    text = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#")).strip()
    _rules_cache = (mtime, text)
    if text:
        logger.info("правила мира перечитаны: %d символов", len(text))
    return text


# Начало промпта, одинаковое для КАЖДОГО NPC и каждой реплики за всю игру.
# Собирается один раз при загрузке модуля, чтобы строка была байт в байт той
# же — иначе переиспользование кеша не сработает.
def _as_int(v: Any) -> Optional[int]:
    """Число из запроса, если оно там есть и это правда число."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


_STATIC_PREFIX = "\n".join([
    "You are roleplaying as an NPC in The Elder Scrolls III: Morrowind.",
    "It is the Third Era, approximately 3E 427, in the province of Morrowind.",
    "",
    "RULES:",
    "- ВСЕГДА отвечай ТОЛЬКО на русском языке — игрок говорит по-русски. (Always answer in Russian, never in English.)",
    "- Respond only in character. Never break the fourth wall.",
    "- Use lore-accurate terminology (in Russian): 'чужак' (outlander), 'н'вах' (n'wah), 'сэра' (sera), 'мутсэра' (muthsera), etc. when appropriate to the character.",
    "- Acknowledge the player's words directly. Be specific, not generic.",
    "- Do not invent lore that contradicts Morrowind canon.",
    "- Before replying, briefly consider what this NPC believes the player wants right now — let that quiet inference shape your tone without stating it aloud.",
    "",
    EMOTION_GUIDE,
    "",
    ACTION_GUIDE,
    "",
    DIRTY_WORK_GUIDE,
    "",
    INSULT_GUIDE,
    "",
    GENDER_GUIDE,
    "",
    PROACTIVE_GUIDE,
    "",
    RESPONSE_SCHEMA,
])


def _build_system_prompt(
    npc_name: str,
    npc_race: str,
    npc_class: str,
    npc_faction: str,
    location: str,
    disposition_band: Optional[str] = None,
    last_mood: Optional[str] = None,
    life_facts: Optional[list[str]] = None,
    talkativeness: str = "normal",
    disposition: Optional[int] = None,
) -> str:
    race_blurb = RACE_PERSONALITIES.get(
        npc_race,
        f"You are a {npc_race}. Roleplay your race appropriately for Morrowind lore.",
    )
    faction_blurb = FACTION_NOTES.get(npc_faction, "")

    # НЕИЗМЕННАЯ ЧАСТЬ ИДЁТ ПЕРВОЙ — это не косметика. И llama.cpp (а значит
    # LM Studio), и облачные модели переиспользуют вычисления для СОВПАДАЮЩЕГО
    # НАЧАЛА промпта. Раньше промпт начинался с имени и настроения NPC, то есть
    # менялся с первой же строки, и кеш не срабатывал никогда: три тысячи
    # токенов руководств пересчитывались на каждую реплику.
    parts = [
        _STATIC_PREFIX,
        "",
        f"NPC NAME: {npc_name}",
        f"NPC RACE: {npc_race}",
        f"NPC CLASS: {npc_class}",
        f"NPC FACTION: {npc_faction or 'None'}",
        f"CURRENT LOCATION: {location}",
        "",
        "PERSONALITY:",
        race_blurb,
    ]

    if faction_blurb:
        parts += ["", "FACTION ROLE:", faction_blurb]

    if life_facts:
        parts += [
            "",
            "PERSONAL BACKGROUND (non-plot; reference naturally if it fits):",
            *[f"- {f}" for f in life_facts[:5]],
        ]

    if disposition_band:
        parts += ["", "RELATIONSHIP:", disposition_band]

    # Настроение прошлой встречи — но только если оно НЕ спорит с тем, как этот
    # человек относится к игроку на самом деле.
    #
    # Настроение пишется из предыдущего ответа самой модели, и получалась петля:
    # ответила холодно -> записалось «disgusted» -> в следующий раз читает «ты
    # испытывал отвращение» -> отвечает ещё холоднее. Игрок вернул Фарготу
    # фамильное кольцо, движок показывал отношение 90 из 100, а тот всё равно
    # цедил сквозь зубы — потому что residue пересиливал число.
    #
    # Источник правды — движок. Он считает отношение по поступкам, а не по
    # тому, каким тоном была прошлая реплика.
    SOUR = {"angry", "disgusted", "fearful"}
    contradicts = (
        last_mood in SOUR and disposition is not None and disposition >= 61
    ) or (
        last_mood == "happy" and disposition is not None and disposition <= 20
    )
    if last_mood and last_mood != "neutral" and not contradicts:
        parts += [
            "",
            f"EMOTIONAL RESIDUE: At your last encounter you felt {last_mood} toward "
            "the player. A quiet echo of that still colours your tone, even if you "
            "try to hide it.",
        ]
    elif contradicts:
        parts += [
            "",
            "EMOTIONAL RESIDUE: none worth mentioning. Whatever the last exchange "
            "sounded like, your actual standing with this person says otherwise — "
            "go by the standing, not by the echo.",
        ]

    # Нелепые судьбы открывает ручка «нелепость»: без этого они выпадали бы и
    # в мире, который игрок настроил серьёзным. Ручка тут не тон задаёт, а
    # разрешение — выбирать всё равно по характеру.
    try:
        import world_tuning as _wt
        _h = _wt.read()["нелепость"]
    except Exception:  # noqa: BLE001
        _h = 0
    if _h >= 25:
        parts += [
            "",
            f"НАСТРОЙ МИРА: нелепость {_h} из 100. Нелепые судьбы "
            "(ticket, actor, prophet, fisher, clerk, guard_ic) РАЗРЕШЕНЫ — "
            "бери их примерно в каждом " + ("втором" if _h >= 60 else "третьем")
            + " случае, когда судьба вообще решается. Проживай такую судьбу "
            "на полном серьёзе: человек не считает её смешной.\n"
            "И ОТДЕЛЬНО ПРОВЕРЬ ПРЯМО СЕЙЧАС: не перевернулась ли твоя жизнь "
            "в этом самом разговоре? Вернули потерянное, выручили в беде, дали "
            "денег в отчаянный час, сдержали слово — FATE:lucky | devotee | "
            "keeper. Обокрали, обманули, разорили, влезли в дом, предали "
            "доверие — FATE:hoarder | sleuth. И то и другое ставится при "
            "ACTION:none: ты никуда не уезжаешь, остаёшься при своих делах, но "
            "с этого дня живёшь иначе. Такое бывает раз в жизни — за обычную "
            "любезность, покупку или ссору судьбу не ставят.",
        ]
    else:
        parts += [
            "",
            "НАСТРОЙ МИРА: серьёзный. Нелепые судьбы (ticket, actor, prophet, "
            "fisher, clerk, guard_ic) НЕ ИСПОЛЬЗУЙ — только worker, innkeep, "
            "drunk, beggar, guard, smuggler.",
        ]

    # Правила, заданные игроком в лаунчере. Ставим ПЕРЕД общими правилами и
    # помечаем как высший приоритет: это его мир и его игра.
    rules = house_rules()
    if rules:
        parts += [
            "",
            "ПРАВИЛА ЭТОГО МИРА (заданы игроком; при расхождении с общими "
            "указаниями выше — ГЛАВНЕЕ они, но канон Morrowind и формат ответа "
            "не отменяются):",
            rules,
        ]

    parts += [
        "",
        "RULES FOR THIS CHARACTER:",
        _TALK_RULES.get(talkativeness, _TALK_RULES["normal"]),
    ]

    return "\n".join(parts)


# Every machine-readable tag the model may emit. Lines starting with any of
# these must never reach the player's screen or the voice synthesizer.
_TAG_PREFIXES = ("EMOTION:", "ACTION:", "TARGET:", "DISP:", "GOLD:",
                 "ITEM:", "HEARD:", "LOAN:", "DEAL:", "COND:", "FATE:")

# Слабые модели путают схему и сочиняют СВОИ служебные строки — замеры на
# локальных моделях дали «EXPECTATION:EMOTION:surprised» и «RESPONSE: …»
# вместо реплики, и это уезжало игроку на экран и в озвучку. Живая русская
# речь никогда не начинается с латинского слова заглавными и двоеточия,
# поэтому любую такую строку считаем служебной, знаем мы этот тег или нет.
_JUNK_TAG_RE = re.compile(r"^[A-Z][A-Z_]{2,}\s*:\s*")
_CYRILLIC_RE = re.compile(r"[а-яёА-ЯЁ]")

# Маркер, выдуманный моделью вместо нашего. В живой игре видели оба вида:
# «<Tidral's Response>» латиницей и «<одунусиус_нуцциус>…</одунусиус_нуцциус>»
# кириллицей, по имени самого NPC. Обе разновидности уезжали игроку в
# субтитры и в озвучку.
#
# Внутри скобок — только буквы, цифры, пробелы и дефисы: ремарка вроде
# «<вздыхает>» тоже попадёт под нож, и правильно — по нашему же формату
# ремарок в реплике быть не должно. А вот «<» в живой фразе («цена < 100»)
# скобкой не закроется и уцелеет.
_MARKER_RE = re.compile(r"</?\s*[\w'’\- ]{1,40}\s*>", re.UNICODE)

# Тег, написанный через равно: своя модель выдавала «LOAN=no», и строка
# уходила в реплику как речь, потому что мы искали «LOAN:».
_TAG_EQ_RE = re.compile(
    r"(?mi)^(\s*)(EMOTION|ACTION|TARGET|DISP|GOLD|ITEM|HEARD|LOAN|DEAL|COND|FATE)"
    r"\s*=\s*")


# Ремарка звёздочками — «*вздыхает*», «*скрестив руки*». Персонаж этого не
# произносит, а синтезатор бы произнёс.
_STAGE_RE = re.compile(r"\*[^*\n]{1,60}\*")


def normalize_tags(raw: str) -> str:
    """Привести теги к нашему виду до разбора: «LOAN=no» -> «LOAN:no»."""
    return _TAG_EQ_RE.sub(lambda m: f"{m.group(1)}{m.group(2).upper()}:", raw or "")


def _despine(line: str) -> str | None:
    """Строка ответа без служебной метки, или None если это чистая служебка.

    Модель может пометить саму реплику («RESPONSE: Ступай мимо») — тогда
    метку снимаем, а речь оставляем. А может выдумать целый тег
    («EXPECTATION:EMOTION:surprised») — такую строку выбрасываем: живой
    речи в ней нет.
    """
    s = _STAGE_RE.sub("", _MARKER_RE.sub("", line)).strip()
    if not s:
        return "" if line.strip() == "" else None
    if s.startswith(_TAG_PREFIXES):
        return None
    m = _JUNK_TAG_RE.match(s)
    if not m:
        return s
    rest = s[m.end():].strip()
    return rest if len(_CYRILLIC_RE.findall(rest)) >= 2 else None


def partial_text(raw: str) -> str:
    """Что показать игроку из НЕДОПИСАННОГО ответа модели.

    Полный разбор здесь не годится: тегов ещё нет, закрывающего маркера тоже.
    Задача скромнее — достать то, что уже произнесено, и не пустить на экран
    служебные строки, даже наполовину набранные.
    """
    text = normalize_tags(raw or "")
    if "<npc_response>" in text:
        text = text.split("<npc_response>", 1)[1]
    if "</npc_response>" in text:
        text = text.split("</npc_response>", 1)[0]
    # Маркер, набранный наполовину («<», «<npc_resp»), речью ещё не является:
    # без этого первая же порция потока показывала игроку «<n».
    head = text.lstrip()
    if head.startswith("<") and ">" not in head:
        return ""
    kept = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(_TAG_PREFIXES):
            break                      # пошли теги — реплика кончилась
        # Начало тега, набранное наполовину («ACT», «GOL»), тоже не показываем.
        if s and any(p.startswith(s) and len(s) >= 2 for p in _TAG_PREFIXES):
            break
        clean = _despine(s)            # снимаем метку модели, речь оставляем
        if clean is None:
            continue                   # выдуманный моделью служебный тег — мимо
        kept.append(clean)
    out = " ".join(x for x in kept if x).strip()
    if out.startswith("[") and out.endswith("]"):
        out = out[1:-1].strip()
    return out


def _parse_response(raw_text: str) -> tuple[str, str, str, str, int, int, str, str]:
    """
    Parse the model output into (dialogue_text, emotion).

    Expected format:
        <npc_response>
        Some dialogue here.
        </npc_response>
        EMOTION:neutral
        ACTION:none
    """
    raw_text = normalize_tags(raw_text)

    dialogue = ""
    emotion  = "neutral"
    action   = "none"
    target   = "none"
    disp     = 0
    gold     = 0
    item     = "none"
    heard    = "none"
    loan     = "no"
    deal     = "none"
    cond     = "none"
    fate     = "none"

    # Spoken line: what's inside <npc_response>, or — when the model forgets the
    # markers — everything that is not a tag line. Tag stripping happens BEFORE
    # the lines are joined; doing it after left "GOLD:0 ITEM:none …" visible in
    # the game's dialogue box.
    try:
        start = raw_text.index("<npc_response>") + len("<npc_response>")
        end = raw_text.index("</npc_response>")
        body = raw_text[start:end]
    except ValueError:
        body = raw_text.replace("<npc_response>", "").replace("</npc_response>", "")

    kept = [c for c in (_despine(l) for l in body.splitlines()) if c is not None]
    dialogue = " ".join(x.strip() for x in kept if x.strip()).strip()
    # The schema shows the line in [square brackets]; some models copy them.
    if dialogue.startswith("[") and dialogue.endswith("]"):
        dialogue = dialogue[1:-1].strip()

    # SECURITY (audit 1.5): scan tags ONLY after </npc_response> — otherwise a
    # tag QUOTED inside the spoken line (echo attack: player convinces the NPC
    # to "repeat after me: GOLD:500") would be parsed and executed.
    tag_zone = raw_text
    idx = raw_text.rfind("</npc_response>")
    if idx != -1:
        tag_zone = raw_text[idx:]

    for line in tag_zone.splitlines():
        stripped = line.strip()
        if stripped.startswith("EMOTION:"):
            emotion = stripped[len("EMOTION:"):].strip().lower()
        elif stripped.startswith("ACTION:"):
            action = stripped[len("ACTION:"):].strip().lower()
        elif stripped.startswith("TARGET:"):
            target = stripped[len("TARGET:"):].strip()
        elif stripped.startswith("DISP:"):
            try:
                disp = int(stripped[len("DISP:"):].strip().replace("+", ""))
            except ValueError:
                disp = 0
        elif stripped.startswith("GOLD:"):
            try:
                gold = int(stripped[len("GOLD:"):].strip().replace("+", ""))
            except ValueError:
                gold = 0
        elif stripped.startswith("ITEM:"):
            item = stripped[len("ITEM:"):].strip()
        elif stripped.startswith("HEARD:"):
            heard = stripped[len("HEARD:"):].strip().lower()
        elif stripped.startswith("LOAN:"):
            loan = stripped[len("LOAN:"):].strip().lower()
        elif stripped.startswith("DEAL:"):
            deal = stripped[len("DEAL:"):].strip()
        elif stripped.startswith("COND:"):
            cond = stripped[len("COND:"):].strip().lower()
        elif stripped.startswith("FATE:"):
            fate = stripped[len("FATE:"):].strip().lower()

    valid_emotions = {"neutral", "happy", "angry", "fearful", "disgusted", "surprised"}
    if emotion not in valid_emotions:
        emotion = "neutral"

    valid_actions = {
        "none", "follow", "flee", "attack", "trade", "callguards", "defend",
        "threaten", "leave", "relocate", "dismiss", "absolve",
        # Тёмные дела — исполняются движком так же по-настоящему, как и прочие.
        "poison", "steal", "plant", "frame", "abduct", "unlock",
        "wait_here", "go_to",
    }
    if action not in valid_actions:
        action = "none"
    if target.lower() in ("none", ""):
        target = "none"
    disp = max(-10, min(10, disp))
    gold = max(-500, min(500, gold))
    if item.lower() in ("none", ""):
        item = "none"
    if heard not in ("none", "alarm"):
        heard = "none"
    if loan not in ("yes", "no"):
        loan = "no"
    if not deal or deal.lower().startswith("none"):
        deal = "none"
    if cond not in ("weapon", "crime", "approach", "theft"):
        cond = "none"
    # Белый список судеб. Держать его в согласии со схемой промпта и с
    # FATE_STORY в Lua ОБЯЗАТЕЛЬНО: модель послушно ставила FATE:lucky, а
    # разбор молча превращал его в none — судьба не начиналась никогда, и со
    # стороны это выглядело так, будто модель не слушается указаний.
    if fate not in FATE_ROLES:
        fate = "none"
    # Strip any stray tag lines that leaked into the dialogue body.
    # (tag lines were already stripped above, before the body was joined)

    return (dialogue, emotion, action, target, disp, gold, item, heard, loan,
            deal, cond, fate)


class LoreAgent:
    """
    Core NPC dialogue agent.

    Generates in-character Morrowind NPC responses using the configured LLM
    provider. Provider and model are read from config['models']['lore_agent'].
    Maintains conversation continuity via ChromaDB memory context passed in
    by the caller.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialise the LoreAgent.

        Args:
            config: Full config dict (as loaded from config.yaml). Must contain
                    a 'models.lore_agent' key with 'provider' and 'model'.
                    Falls back to gemini-2.5-flash if the key is missing.
        """
        provider_cfg: dict = config.get("models", {}).get(
            "lore_agent", {"provider": "gemini", "model": "gemini-2.5-flash"}
        )
        self.llm = get_provider(provider_cfg)
        self._temperature: float = provider_cfg.get("temperature", 0.85)
        self._max_tokens: int = config.get("max_output_tokens", 200)
        logger.info(
            "LoreAgent initialised: provider=%s model=%s",
            provider_cfg.get("provider"),
            provider_cfg.get("model"),
        )

    async def generate_response(
        self,
        request: dict[str, Any],
        memory_context: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Generate an in-character NPC response.

        Args:
            request: Dict with keys:
                - npc_id (str)
                - npc_name (str)
                - npc_race (str)
                - npc_class (str)
                - npc_faction (str)
                - player_input (str)
                - location (str)
                - conversation_history (list of {role, content} dicts)  [optional]
            memory_context: List of memory dicts from ChromaDB retrieval,
                each with at least a 'content' key.

        Returns:
            {
                "response": str,      # in-character dialogue
                "emotion": str,       # detected emotion tag
                "tokens_used": int,   # total tokens consumed
                "cost_usd": float,    # estimated USD cost
            }
        """
        npc_name = request.get("npc_name", "Stranger")
        npc_race = request.get("npc_race", "Dunmer")
        npc_class = request.get("npc_class", "Commoner")
        npc_faction = request.get("npc_faction", "")
        location = request.get("location", "Vvardenfell")
        player_input = request.get("player_input", "")
        is_greeting: bool = request.get("is_greeting", player_input == "")
        conversation_history: list[dict] = request.get("conversation_history", [])

        # Optional disposition context injected by the bridge. When the feature
        # flag is off these are all None and the prompt is unchanged.
        disposition_band = request.get("disposition_band")
        last_mood        = request.get("last_mood")
        life_facts       = request.get("life_facts") or []
        player_context   = (request.get("player_context") or "").strip()
        active_quests    = (request.get("active_quests") or "").strip()
        rumors           = request.get("rumors") or []
        bystanders       = (request.get("bystanders") or "").strip()
        corpses          = (request.get("corpses") or "").strip()
        npc_place        = (request.get("npc_place") or "").strip()
        npc_fate         = (request.get("npc_fate") or "").strip()
        npc_drives       = (request.get("npc_drives") or "").strip()
        risk_note        = (request.get("risk_note") or "").strip()
        npc_condition    = (request.get("npc_condition") or "").strip()

        system_prompt = _build_system_prompt(
            npc_name=npc_name,
            npc_race=npc_race,
            npc_class=npc_class,
            npc_faction=npc_faction,
            location=location,
            disposition_band=disposition_band,
            last_mood=last_mood,
            life_facts=life_facts,
            talkativeness=str(request.get("talkativeness") or "normal"),
            # Отношение из движка — чтобы прошлое настроение не спорило с тем,
            # как этот человек относится к игроку на самом деле.
            disposition=_as_int(request.get("npc_disposition")),
        )

        # Build the user turn, prepending memory context if available
        user_parts: list[str] = []

        baked = (request.get("baked_traits") or "").strip()
        if baked:
            user_parts.append(
                "ТВОЙ ВРОЖДЁННЫЙ ХАРАКТЕР (неизменен всю игру — это ты и есть, "
                "в любой день и при любой встрече; отыгрывай ПОСЛЕДОВАТЕЛЬНО, "
                "особенно отношение к деньгам):\n" + baked
            )

        # Стражник пришёл на вызов. До этого «позвать стражу» означало
        # мгновенный штраф игроку — закон срабатывал раньше, чем кто-либо
        # разобрался, кто прав, и пяти выкриков подряд хватало, чтобы сесть.
        inquiry = str(request.get("inquiry") or "").strip()
        if inquiry:
            caller, _, starter = inquiry.partition("|")
            user_parts.append(
                "ТЫ СТРАЖНИК И ПРИШЁЛ НА ВЫЗОВ. Драку ты уже разнял, оружие "
                "велел убрать. Теперь разбираешься.\n"
                f"Кто звал: {caller or 'неизвестно'}.\n"
                f"Кто начал: {starter or 'неизвестно'}.\n"
                "\n"
                "Ты НЕ ЗНАЕШЬ, кто прав, пока не спросишь. Начни с вопроса — "
                "к игроку, к заявителю, к тем, кто рядом. Никого не бей и "
                "никого не штрафуй, пока не разобрался: закон здесь ты, а не "
                "чужой крик.\n"
                "\n"
                "Когда решишь:\n"
                "- виноват игрок — ACTION:callguards (это твой приговор: штраф "
                "и под стражу);\n"
                "- виноват заявитель или оба хороши — ACTION:none, и скажи это "
                "вслух своими словами;\n"
                "- разбираться неохота — тоже ACTION:none: «разошлись оба, пока "
                "я добрый», и дело с концом.\n"
                "\n"
                "Отыгрывай СЕБЯ, а не устав: усталому стражнику под конец смены "
                "лень возиться, дотошный будет допрашивать до мелочей, а "
                "продажный намекнёт, что дело можно уладить и без бумаг."
            )

        # Спутник обязан помнить, что он спутник. Без этой строки NPC шёл за
        # игроком по пятам и в том же разговоре отрицал сам факт, заявляя, что
        # это игрок за ним увязался.
        if request.get("is_companion"):
            user_parts.append(
                "ТЫ ПУТЕШЕСТВУЕШЬ С ЭТИМ ЧЕЛОВЕКОМ. Ты сам согласился идти "
                "следом и идёшь за ним прямо сейчас — это факт, а не его "
                "выдумка. Не отрицай его, не переворачивай («это ты за мной "
                "увязался») и не делай вид, что вы встретились случайно. "
                "Захотел уйти — скажи об этом прямо, словами."
            )

        npc_disposition = request.get("npc_disposition")
        if npc_disposition is not None:
            user_parts.append(
                f"YOUR DISPOSITION toward the player on the game's 0-100 scale: {npc_disposition} "
                "(0-20 hostile contempt, 21-40 wary dislike, 41-60 neutral, "
                "61-80 friendly, 81-100 trusting warmth). Let it colour your tone."
            )

        if player_context:
            user_parts.append(
                "CURRENT SCENE (what this NPC can see right now — react naturally "
                "to anything notable: drawn weapons, wounds, odd hour, storms, "
                "the player's bounty or sickness):\n" + player_context
            )

        if npc_condition:
            user_parts.append(
                "YOUR OWN CONDITION right now (roleplay it — a dying character "
                "may beg for mercy, offer gold, betray accomplices; a sick one "
                "coughs and drifts):\n" + npc_condition
            )

        if risk_note:
            user_parts.append(
                "WHAT IT WOULD COST YOU (real, from the scene around you):\n"
                + risk_note + "\n"
                "Any crime the player proposes must be weighed against THIS, not "
                "against the size of the purse alone. Guards at your shoulder or a "
                "room full of witnesses make a bribe laughable — say so plainly. "
                "An empty street makes the same offer worth considering. And a "
                "stranger with a heavy bounty is himself a danger to be seen with."
            )

        if npc_drives:
            user_parts.append(
                "WHAT YOU WANT FOR YOURSELF (your own life, quite apart from "
                "this stranger):\n" + npc_drives + "\n"
                "Weigh every offer against it: a proposal that serves your goal "
                "tempts you even when it is unsavoury, one that endangers it you "
                "refuse however well it pays. When the talk allows, steer it "
                "toward what you need — ask for help, hint at your trouble, "
                "propose a bargain of your own. Do not recite this list; live it."
            )

        arc = request.get("companion_arc") or []
        arc_reveal = int(request.get("arc_reveal") or 1)
        if arc:
            opened = [s for s in arc[:max(1, min(len(arc), arc_reveal))]]
            hidden = arc[len(opened):]
            block = ["YOUR OWN HIDDEN STORY as this player's travelling companion.",
                     "Already surfaced (you may speak of this, in your own time and "
                     "on your own terms — not as an info-dump):"]
            block += [f"  - {s}" for s in opened]
            if hidden:
                block.append("STILL SEALED — you know it, but you do NOT tell it yet. "
                             "Deflect, half-lie, change the subject. It opens only as "
                             "the player's own story goes further:")
                block += [f"  - (закрыто) {s}" for s in hidden]
            block.append("Bring it up yourself when the place, the danger or the talk "
                         "touches it — that is what makes travelling with you matter.")
            user_parts.append("\n".join(block))

        if npc_fate:
            user_parts.append(
                "WHAT LIFE HAS DONE WITH YOU SINCE (this is your real situation "
                "now — speak from it, and let it change how you greet the "
                "player: gratitude, shame, resentment, pride):\n" + npc_fate + "\n"
                "If you owe them and you have prospered, PAY THEM BACK with "
                "interest in this very line (negative GOLD) — a debt repaid "
                "handsomely is how a decent person thanks the one who gave them "
                "a start. If you squandered it, do not pretend otherwise."
            )

        if npc_place:
            user_parts.append(
                "WHERE YOU ARE STANDING (this is YOUR situation, not the "
                "player's — never claim to be somewhere else):\n" + npc_place
            )

        if corpses:
            user_parts.append(
                "WHAT LIES IN FRONT OF YOU RIGHT NOW:\n" + corpses + "\n"
                "You cannot small-talk over a body as if the room were empty. "
                "If the note says YOU killed them, you know it and you own it — "
                "denying it or asking where the culprit went is impossible."
            )

        if bystanders:
            user_parts.append(
                "WHO ELSE CAN HEAR THIS CONVERSATION:\n" + bystanders + "\n"
                "Mind them ONLY when the topic is genuinely sensitive (stolen goods, "
                "smuggling, conspiracies, slander of the powerful): then hint, deflect "
                "or whisper. In ORDINARY small talk do NOT mention the listeners or "
                "guards at all — constant paranoia looks ridiculous."
            )

        if active_quests:
            user_parts.append(
                "ДЕЛА, О КОТОРЫХ ТЫ МОЖЕШЬ ЗНАТЬ — это записи журнала игры "
                "своими словами, ты назван в них поимённо, поэтому говори о них "
                "как о своих собственных делах:\n"
                + active_quests + "\n"
                # Про сделанное приходилось напоминать отдельно: раньше сюда
                # попадали только НЕЗАКОНЧЕННЫЕ дела, и человек не помнил
                # услуги, которую ему оказали час назад.
                "Всё, что помечено как СДЕЛАННОЕ, — уже случилось на самом деле. "
                "Это твой долг перед этим человеком, а не слухи и не его "
                "хвастовство: не переспрашивай, не сомневайся вслух и не проси "
                "сделать это ещё раз. Если он о таком заговорит — ты помнишь и "
                "отвечаешь как человек, которому помогли.\n"
                "Чего в списке нет — того ты не знаешь: не выдавай чужих "
                "поручений и никогда не перечисляй список без спросу."
            )

        deal_note = (request.get("deal_note") or "").strip()
        if deal_note:
            user_parts.append(
                "ДЕЙСТВУЮЩИЙ УГОВОР С ИГРОКОМ (движок его исполняет — говори о нём "
                "как о реальном деле, не предлагай новый):\n" + deal_note
            )

        debt_note = (request.get("debt_note") or "").strip()
        if debt_note:
            user_parts.append(
                "ДОЛГ (реальный, движок его считает — не выдумывай сумм):\n" + debt_note
            )

        npc_inventory = (request.get("npc_inventory") or "").strip()
        if npc_inventory:
            user_parts.append(
                "WHAT YOU CARRY (your actual inventory — you may hand over or "
                "mention ONLY these things; ITEM must match one of them):\n"
                + npc_inventory
            )

        npc_canon = (request.get("npc_canon") or "").strip()
        if npc_canon:
            user_parts.append(
                "WHAT THIS NPC ACTUALLY SAYS IN THE GAME (their personal vanilla "
                "dialogue lines from the game data — YOUR SOURCE OF TRUTH about "
                "this character's knowledge, business and quests; stay consistent "
                "with it, never contradict it, never invent quests beyond it; "
                "the [tag] is the topic name):\n" + npc_canon
            )

        if rumors:
            user_parts.append(
                "RUMORS you heard through gossip (secondhand! retell IMPRECISELY — "
                "gossip drifts: names blur, numbers grow, details mutate; if it "
                "happened far from here you may barely know it; weave one in ONLY "
                "if fitting, don't recite):\n"
                + "\n".join(f"- {r}" for r in rumors)
            )

        if memory_context:
            mem_lines = []
            for entry in memory_context[:5]:  # cap at 5 memories
                content = entry.get("content", "")
                if content:
                    mem_lines.append(f"- {content}")
            if mem_lines:
                user_parts.append(
                    "RELEVANT MEMORIES (what this NPC knows from past interactions):\n"
                    + "\n".join(mem_lines)
                )

        if conversation_history:
            history_lines = []
            for turn in conversation_history[-6:]:  # last 3 exchanges
                role = turn.get("role", "?")
                content = turn.get("content", "")
                history_lines.append(f"{role.upper()}: {content}")
            if history_lines:
                user_parts.append(
                    "RECENT CONVERSATION:\n" + "\n".join(history_lines)
                )

        death_react = (request.get("death_react") or "").strip()
        theft_item = (request.get("theft_item") or "").strip()
        if death_react:
            user_parts.append(
                f"ТОЛЬКО ЧТО ПОБЛИЗОСТИ ПОГИБ(ЛА): «{death_react}». Ты это узнал/видел. "
                "Реагируй по СВОИМ отношениям с погибшим: сослуживец или сосед — горе, "
                "страх, гнев; недруг — плохо скрытое злорадство; никто — тревожное "
                "любопытство. Реши, виновен ли игрок (он рядом; смотри сцену — оружие "
                "наголо? штраф?): обвиняй, требуй объяснений, зови закон (ACTION:callguards), "
                "а мстительный и уверенный может напасть (ACTION:attack). Не будь безразличным."
            )
        elif theft_item:
            user_parts.append(
                f"ПРЯМО СЕЙЧАС, У ТЕБЯ НА ГЛАЗАХ, игрок взял ТВОЮ вещь: «{theft_item}». "
                "Это твоя собственность. Реагируй по характеру: возмутись и потребуй "
                "вернуть; пригрози (ACTION:threaten); позови стражу (ACTION:callguards); "
                "вспыльчивый может кинуться в драку (ACTION:attack); трус лишь процедит "
                "сквозь зубы — но запомнит (DISP сильно вниз в любом случае)."
            )
        elif request.get("is_surrender"):
            user_parts.append(
                "БОЙ: ты только что бился с игроком НАСМЕРТЬ и ПРОИГРЫВАЕШЬ — ты на грани "
                "гибели и сам опустил оружие. Реши по своему характеру, что делать: "
                "молить о пощаде; предложить откуп (золото, тайник, вещь); сдать подельников "
                "или ценные сведения; поклясться уйти и начать честную жизнь; попытаться "
                "обмануть и выиграть время — или гордо плюнуть и драться до конца "
                "(тогда ACTION:attack — бой возобновится). Говори отчаянно, коротко, по делу."
            )
        elif request.get("is_proactive"):
            user_parts.append(
                "YOU have decided to address the player FIRST, right now — they did "
                "not approach you. Pick a concrete reason from your context (a rumor "
                "you heard about them, their bounty or sickness, their gear, your "
                "trade or duty, your own troubles) and hail them with it: a greeting, "
                "a warning, an offer, a plea for help, a veiled threat — whatever "
                "fits YOUR character. Make it specific, not generic small talk."
            )
        elif is_greeting:
            user_parts.append(
                "The player has just approached and made eye contact. "
                "Greet them in character — a short, natural opening line appropriate to this NPC's personality."
            )
        else:
            user_parts.append(f"PLAYER SAYS: {player_input}")
        user_message = "\n\n".join(user_parts)

        messages = [{"role": "user", "content": user_message}]

        # Потоковая выдача: реплика в нашем формате идёт ПЕРВОЙ, а теги действий
        # в конце — поэтому текст можно показывать по мере набора, ничем не
        # рискуя: мир меняется только по готовым тегам.
        on_partial = request.get("on_partial")
        if on_partial is not None and getattr(self.llm, "supports_stream", False):
            resp = await call_with_retry(
                lambda: self.llm.complete_stream(
                    system=system_prompt,
                    messages=messages,
                    on_text=lambda raw: on_partial(partial_text(raw)),
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
            )
        else:
            resp = await call_with_retry(
                lambda: self.llm.complete(
                    system=system_prompt,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
            )

        (dialogue, emotion, action, target, disp, gold, item, heard, loan,
         deal, cond, fate) = _parse_response(resp.text)
        total_tokens = resp.tokens_in + resp.tokens_out

        log_llm_response("LoreAgent", resp)

        logger.debug(
            "LoreAgent | npc=%s | tokens=%d | cost=$%.5f | emotion=%s | action=%s",
            npc_name, total_tokens, resp.cost_usd, emotion, action,
        )

        return {
            "response": dialogue,
            "emotion": emotion,
            "action": action,
            "target": target,
            "disp": disp,
            "gold": gold,
            "item": item,
            "heard": heard,
            "loan": loan,
            "deal": deal,
            "cond": cond,
            "fate": fate,
            "tokens_used": total_tokens,
            "cost_usd": resp.cost_usd,
        }

    async def generate_companion_arc(
        self,
        npc_name: str,
        npc_race: str,
        npc_class: str,
        npc_faction: str,
        npc_canon: str = "",
        quests: str = "",
    ) -> list[str]:
        """A hidden personal story for someone the player takes along.

        Three stages, from a hint to the whole truth, tied to the war between
        the Tribunal, the Sixth House and the Empire — the fabric of the main
        quest — but never rewriting it. Generated ONCE and kept in the savegame,
        revealed as the player's own story advances.
        """
        race_blurb = RACE_PERSONALITIES.get(npc_race, f"A {npc_race} of Morrowind.")
        system = (
            "Ты придумываешь СКРЫТУЮ ЛИЧНУЮ ИСТОРИЮ спутника для ролевой игры "
            "по Morrowind (3Э 427). ПИШИ ТОЛЬКО ПО-РУССКИ.\n"
            "Три строки — три ступени раскрытия, от намёка к полной правде:\n"
            "1) странность, которую спутник выдаёт случайно (оговорка, привычка, "
            "страх, вещь, которую он прячет);\n"
            "2) полупризнание под нажимом обстоятельств;\n"
            "3) вся правда — кем он был и чего хочет на самом деле.\n"
            "Связь с большой историей провинции обязательна: Шестой Дом и "
            "шёпот Дагот Ура, Трибунал и Храм, имперские сборщики податей, "
            "Мораг Тонг, работорговцы, Великие Дома. Он может быть беглым "
            "послушником, бывшим осведомителем, родичем пропавшего, носителем "
            "проклятия — но НИКОГДА не переписывай канон: он не тайный Нереварин, "
            "не Вивек и не Дагот Ур, и он не отменяет ни одного квеста.\n"
            "Каждая строка — одно предложение. Без нумерации и пояснений."
        )
        user_parts = [
            f"СПУТНИК: {npc_name}", f"РАСА: {npc_race}", f"КЛАСС: {npc_class}",
            f"ФРАКЦИЯ: {npc_faction or 'нет'}", "", f"О расе: {race_blurb}",
        ]
        if npc_canon:
            user_parts += ["", "ЕГО КАНОННЫЕ РЕПЛИКИ (не противоречить им):",
                           npc_canon[:600]]
        if quests:
            user_parts += ["", f"ГДЕ СЕЙЧАС ИГРОК ПО СЮЖЕТУ: {quests[:200]}"]
        user_parts += ["", "Верни ровно три строки и ничего больше."]

        try:
            resp = await call_with_retry(
                lambda: self.llm.complete(
                    system=system,
                    messages=[{"role": "user", "content": "\n".join(user_parts)}],
                    temperature=0.95, max_tokens=320,
                )
            )
            lines = [l.strip(" -–—•*0123456789.").strip()
                     for l in (resp.text or "").splitlines()]
            return [l for l in lines if len(l) > 12][:3]
        except Exception as exc:  # noqa: BLE001
            logger.warning("companion arc gen failed for %s: %s", npc_name, exc)
            return []

    async def generate_life_facts(
        self,
        npc_name: str,
        npc_race: str,
        npc_class: str,
        npc_faction: str,
    ) -> list[str]:
        """
        One-shot: invent 3 short non-plot life facts for this NPC.

        Cached forever in DispositionStore once returned. Small models sometimes
        wrap output in prose — we strip bullets and keep the first 3 useful lines.
        """
        race_blurb = RACE_PERSONALITIES.get(npc_race, f"A {npc_race} of Morrowind.")
        faction_blurb = FACTION_NOTES.get(npc_faction, "")

        system = (
            "You invent personal colour for a Morrowind NPC. "
            "ПИШИ ТОЛЬКО ПО-РУССКИ (the facts are injected into a Russian-language "
            "roleplay prompt — English lines break the illusion). "
            "Output THREE short life facts, one per line, no numbering, no prose. "
            "Each fact is 1 sentence, NON-plot, NON-quest, mundane and human: a "
            "dead sister, a coin collection, a fear of cliff racers, a grudge "
            "against a neighbour. Grounded in Morrowind (3E 427). Avoid clichés."
        )
        user_parts = [
            f"NPC: {npc_name}",
            f"RACE: {npc_race}",
            f"CLASS: {npc_class}",
            f"FACTION: {npc_faction or 'None'}",
            "",
            f"Race note: {race_blurb}",
        ]
        if faction_blurb:
            user_parts.append(f"Faction note: {faction_blurb}")
        user_parts.append("")
        user_parts.append("Return exactly three lines of life facts and nothing else.")
        user = "\n".join(user_parts)

        try:
            resp = await call_with_retry(
                lambda: self.llm.complete(
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    temperature=0.95,
                    max_tokens=160,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("life_facts generation failed for %s: %s", npc_name, exc)
            return []

        log_llm_response("LoreAgent.life_facts", resp)

        facts: list[str] = []
        for raw in (resp.text or "").splitlines():
            line = raw.strip().lstrip("-*•0123456789.) ").strip()
            if len(line) >= 6 and not line.lower().startswith(("npc:", "race:", "class:", "faction:")):
                facts.append(line)
            if len(facts) >= 3:
                break
        return facts
