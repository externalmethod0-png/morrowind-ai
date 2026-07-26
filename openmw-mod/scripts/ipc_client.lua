-- ipc_client.lua (GLOBAL)
-- Receives dialogue requests from PLAYER script; sends via print-IPC;
-- polls VFS response files; relays replies back to the player.
--
-- Request path (Lua -> Python):
--   print('[MWAI_REQ] <json>') — tailed from openmw.log by openmw_log_bridge.py
--
-- Response path (Python -> Lua):
--   VFS: ai_inbox/response.txt    (player dialogue replies)
--   VFS: ai_inbox/npc_speech.txt  (radiant D2D ambient lines)
--   Requires data=C:\morrowind-ai-mod in openmw.cfg (Windows) or
--            data=/mnt/c/morrowind-ai-mod (Linux WSL).
--
-- OpenMW 0.49, Lua 5.1

local core  = require('openmw.core')
local world = require('openmw.world')
local json  = require('scripts.json')

-- Optional VFS (0.49+). Falls back to io.open for Linux.
local vfs    = nil
local io_lib = io or nil

local ok_vfs = pcall(function() vfs = require('openmw.vfs') end)
if not ok_vfs then vfs = nil end

-- ── Paths ─────────────────────────────────────────────────────────────────────

-- VFS paths (relative to data= root, i.e. C:\morrowind-ai-mod\ or equivalent)
local RESPONSE_VFS    = 'ai_inbox/response.txt'
local NPC_SPEECH_VFS  = 'ai_inbox/npc_speech.txt'

-- Linux io.open fallback paths
local RESPONSE_IO     = '/home/nemoclaw/morrowind-ai/ipc/response.json'

local POLL_INTERVAL   = 0.25   -- seconds between VFS polls

-- ── State ─────────────────────────────────────────────────────────────────────

local pollTimer          = 0
local lastRespReqId      = ''
local lastSpeechReqId    = ''

-- Speech queue: list of formatted strings, displayed one at a time.
local speechQueue        = {}
local speechTimer        = 0
local SPEECH_DISPLAY_GAP = 3.5   -- seconds between ambient NPC lines

-- ── Helpers ───────────────────────────────────────────────────────────────────

local function vfs_read(path)
    if vfs then
        local ok, content = pcall(function()
            if not vfs.fileExists(path) then return nil end
            local f = vfs.open(path)
            if not f then return nil end
            local text = f:read('*a')
            f:close()
            return text
        end)
        if ok and content and content ~= '' then return content end
    end
    return nil
end

local function io_read(path)
    if not io_lib then return nil end
    local content
    pcall(function()
        local f = io_lib.open(path, 'r')
        if not f then return end
        content = f:read('*a')
        f:close()
    end)
    return content
end

local function read_response_file()
    return vfs_read(RESPONSE_VFS) or io_read(RESPONSE_IO)
end

local function get_player()
    local p
    pcall(function()
        if world.players and world.players[1] then p = world.players[1] end
    end)
    return p
end

-- ── Request sender ────────────────────────────────────────────────────────────

local reqCounter = 0

local function sendRequest(data)
    reqCounter = reqCounter + 1
    data.req_id = data.req_id or ('req-' .. tostring(math.floor(core.getSimulationTime())) .. '-' .. reqCounter)
    local ok, enc = pcall(json.encode, data)
    if ok then
        print('[MWAI_REQ] ' .. enc)
        print('[morrowind-ai] Sent ' .. tostring(data.type) .. ' req_id=' .. data.req_id)
    else
        print('[morrowind-ai] JSON encode error: ' .. tostring(enc))
    end
    return data.req_id
end

-- ── Event handlers ────────────────────────────────────────────────────────────

local function onDialogueRequest(data)
    if not data then return end
    sendRequest({
        type        = 'dialogue',
        npc_id      = tostring(data.npc_id      or ''),
        npc_name    = tostring(data.npc_name    or ''),
        npc_race    = tostring(data.npc_race    or ''),
        npc_class   = tostring(data.npc_class   or ''),
        npc_faction = tostring(data.npc_faction or ''),
        location    = tostring(data.location    or ''),
        player_text = tostring(data.player_text or ''),
    })
end

local function onLockNpc(data)
    if not data then return end
    sendRequest({
        type        = 'lock_npc',
        npc_id      = tostring(data.npc_id      or ''),
        npc_name    = tostring(data.npc_name    or ''),
        npc_race    = tostring(data.npc_race    or ''),
        npc_class   = tostring(data.npc_class   or ''),
        npc_faction = tostring(data.npc_faction or ''),
        location    = tostring(data.location    or ''),
    })
end

-- ── Polling: dialogue response ─────────────────────────────────────────────

local function pollDialogueResponse()
    local content = read_response_file()
    if not content or content == '' then return end

    local ok, decoded = pcall(json.decode, content)
    if not ok or type(decoded) ~= 'table' then return end

    local rid = tostring(decoded.req_id or '')
    if rid == '' or rid == lastRespReqId then return end
    lastRespReqId = rid

    local text = tostring(decoded.npc_response or decoded.response or '')
    if text == '' then return end

    local player = get_player()
    if player then
        player:sendEvent('MorrowindAiDialogueReply', {
            npc_id       = tostring(decoded.npc_id or ''),
            npc_response = text,
            emotion      = tostring(decoded.emotion or 'neutral'),
            action       = tostring(decoded.action  or 'none'),
        })
    end
end

-- ── Polling: NPC ambient speech (D2D) ─────────────────────────────────────

local function pollNpcSpeech()
    local content = vfs_read(NPC_SPEECH_VFS)
    if not content or content == '' then return end

    local ok, decoded = pcall(json.decode, content)
    if not ok or type(decoded) ~= 'table' then return end

    local rid = tostring(decoded.req_id or '')
    if rid == '' or rid == lastSpeechReqId then return end
    lastSpeechReqId = rid

    local exchanges = decoded.exchanges
    if type(exchanges) ~= 'table' then return end

    for _, ex in ipairs(exchanges) do
        local name = tostring(ex.speaker_name or '?')
        local text = tostring(ex.text or '')
        if text ~= '' then
            speechQueue[#speechQueue + 1] = '[' .. name .. '] ' .. text
        end
    end
end

-- ── Update ─────────────────────────────────────────────────────────────────

local function onUpdate(dt)
    -- Poll response files
    pollTimer = pollTimer + dt
    if pollTimer >= POLL_INTERVAL then
        pollTimer = 0
        pollDialogueResponse()
        pollNpcSpeech()
    end

    -- Drain speech queue one line at a time with gap
    if #speechQueue > 0 then
        speechTimer = speechTimer + dt
        if speechTimer >= SPEECH_DISPLAY_GAP then
            speechTimer = 0
            local line = table.remove(speechQueue, 1)
            local player = get_player()
            if player then
                player:sendEvent('MorrowindAiNpcSpeech', { text = line })
            end
        end
    end
end

print('[morrowind-ai][ipc_client] Loaded. vfs=' .. tostring(vfs ~= nil) ..
      ' io=' .. tostring(io_lib ~= nil))

return {
    engineHandlers = { onUpdate = onUpdate },
    eventHandlers  = {
        MorrowindAiDialogueRequest = onDialogueRequest,
        MorrowindAiLockNpc         = onLockNpc,
    },
}
