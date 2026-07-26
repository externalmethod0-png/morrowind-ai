-- npc_detector.lua
-- GLOBAL script.
-- Scans all active actors every second (throttled).
-- When two named NPCs are within PROXIMITY_THRESHOLD units and cooldown has
-- elapsed, fires a [MWAI_REQ] print line that openmw_log_bridge.py tails.
--
-- Windows-compatible: uses print() IPC, no io.open.
-- OpenMW 0.49, Lua 5.1

local core  = require('openmw.core')
local world = require('openmw.world')
local types = require('openmw.types')
local json  = require('scripts.json')

-- ============================================================
-- Configuration
-- ============================================================

local SCAN_INTERVAL       = 5.0    -- seconds between scans
local PROXIMITY_THRESHOLD = 300    -- engine units (npc<->npc)
local PLAYER_RADIUS       = 1200   -- only chatter the player can actually see/hear
local PAIR_COOLDOWN       = 600.0  -- seconds before the same pair fires again
local GLOBAL_COOLDOWN     = 90.0   -- seconds between ANY two radiant events
local REQ_COUNTER         = 0      -- incrementing req_id
local lastGlobalFire      = -math.huge

-- ============================================================
-- State
-- ============================================================

local scanTimer  = 0
local pairTimers = {}  -- "id_a|id_b" -> last trigger time

-- ============================================================
-- Helpers
-- ============================================================

local function distance(posA, posB)
    local dx = posA.x - posB.x
    local dy = posA.y - posB.y
    local dz = posA.z - posB.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)
end

local function pairKey(idA, idB)
    return idA < idB and (idA .. '|' .. idB) or (idB .. '|' .. idA)
end

local function isNamedNpc(actor)
    local ok, result = pcall(function()
        if not types.NPC.objectIsInstance(actor) then return false end
        if types.Player.objectIsInstance(actor) then return false end  -- never pair the player
        local rec = types.NPC.record(actor)
        return (rec.name or '') ~= ''
    end)
    return ok and result
end

local function npcInfo(actor)
    local info = { id = '', name = '', race = '', faction = '', position = nil }
    pcall(function()
        info.id = tostring(actor.id or actor.recordId or '')
        local rec = types.NPC.record(actor)
        info.name    = tostring(rec.name  or '')
        info.race    = tostring(rec.race  or 'Unknown')
        info.faction = tostring(rec.faction or '')
        info.position = actor.position
    end)
    return info
end

-- ============================================================
-- Core scan logic
-- ============================================================

local function scanActors()
    local npcs = {}
    local ok, err = pcall(function()
        for _, actor in ipairs(world.activeActors) do
            if isNamedNpc(actor) then
                local info = npcInfo(actor)
                if info.id ~= '' and info.position then
                    npcs[#npcs + 1] = info
                end
            end
        end
    end)
    if not ok then
        print('[npc_detector] scan error: ' .. tostring(err))
        return
    end

    local now = core.getSimulationTime()
    if (now - lastGlobalFire) < GLOBAL_COOLDOWN then return end

    local location = ''
    local playerPos = nil
    pcall(function()
        local player = world.players and world.players[1]
        if player then
            playerPos = player.position
            if player.cell then location = tostring(player.cell.name or '') end
        end
    end)

    for i = 1, #npcs do
        for j = i + 1, #npcs do
            local a, b = npcs[i], npcs[j]
            if a.position and b.position then
                local dist = distance(a.position, b.position)
                local nearPlayer = playerPos ~= nil
                    and distance(a.position, playerPos) <= PLAYER_RADIUS
                if dist <= PROXIMITY_THRESHOLD and nearPlayer then
                    local key = pairKey(a.id, b.id)
                    local lastTime = pairTimers[key] or -math.huge
                    if (now - lastTime) >= PAIR_COOLDOWN then
                        pairTimers[key] = now
                        lastGlobalFire = now
                        REQ_COUNTER = REQ_COUNTER + 1
                        local payload = {
                            type         = 'npc_npc',
                            req_id       = 'npc_npc-' .. tostring(math.floor(now)) .. '-' .. REQ_COUNTER,
                            npc_a_id     = a.id,
                            npc_a_name   = a.name,
                            npc_a_race   = a.race,
                            npc_a_faction = a.faction,
                            npc_b_id     = b.id,
                            npc_b_name   = b.name,
                            npc_b_race   = b.race,
                            npc_b_faction = b.faction,
                            location     = location,
                            timestamp    = now,
                        }
                        local encOk, encoded = pcall(json.encode, payload)
                        if encOk then
                            print('[MWAI_REQ] ' .. encoded)
                            print(string.format('[npc_detector] D2D fired: %s + %s (%.1f units)',
                                a.name, b.name, dist))
                        else
                            print('[npc_detector] JSON encode error: ' .. tostring(encoded))
                        end
                        -- one event per scan to avoid log spam
                        return
                    end
                end
            end
        end
    end
end

-- ============================================================
-- Script engine interface
-- ============================================================

local function onUpdate(dt)
    scanTimer = scanTimer + dt
    if scanTimer < SCAN_INTERVAL then return end
    scanTimer = 0
    scanActors()
end

print('[morrowind-ai][npc_detector] Loaded (print-IPC, Windows-compatible).')

return {
    engineHandlers = { onUpdate = onUpdate },
}
