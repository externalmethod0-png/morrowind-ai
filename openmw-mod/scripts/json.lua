-- json.lua
-- Minimal JSON encoder/decoder for OpenMW Lua (Lua 5.1 compatible)
-- No external dependencies.

local json = {}

-- ============================================================
-- Encoder
-- ============================================================

local function escape_str(s)
    s = s:gsub('\\', '\\\\')
    s = s:gsub('"',  '\\"')
    s = s:gsub('\n', '\\n')
    s = s:gsub('\r', '\\r')
    s = s:gsub('\t', '\\t')
    -- Escape control characters (Lua 5.1: no \xHH in patterns)
    s = s:gsub('[%z\1-\31\127]', function(c)
        return string.format('\\u%04x', string.byte(c))
    end)
    return s
end

local encode  -- forward declaration

local function encode_value(val)
    local t = type(val)
    if val == nil or val == json.null then
        return 'null'
    elseif t == 'boolean' then
        return tostring(val)
    elseif t == 'number' then
        if val ~= val then return 'null' end  -- NaN
        if val == math.huge or val == -math.huge then return 'null' end
        -- Use integer format when value is a whole number
        if val == math.floor(val) and math.abs(val) < 2^53 then
            return string.format('%d', val)
        end
        return string.format('%.14g', val)
    elseif t == 'string' then
        return '"' .. escape_str(val) .. '"'
    elseif t == 'table' then
        return encode(val)
    else
        error('json.encode: unsupported type: ' .. t)
    end
end

-- Detect if a table is an array (consecutive integer keys starting at 1)
local function is_array(t)
    local max = 0
    local count = 0
    for k, _ in pairs(t) do
        if type(k) ~= 'number' or k ~= math.floor(k) or k < 1 then
            return false
        end
        if k > max then max = k end
        count = count + 1
    end
    return count == max
end

encode = function(tbl)
    if is_array(tbl) then
        local parts = {}
        for i = 1, #tbl do
            parts[i] = encode_value(tbl[i])
        end
        return '[' .. table.concat(parts, ',') .. ']'
    else
        local parts = {}
        for k, v in pairs(tbl) do
            if type(k) ~= 'string' then
                error('json.encode: object keys must be strings, got: ' .. type(k))
            end
            parts[#parts + 1] = '"' .. escape_str(k) .. '":' .. encode_value(v)
        end
        return '{' .. table.concat(parts, ',') .. '}'
    end
end

function json.encode(val)
    return encode_value(val)
end

-- ============================================================
-- Decoder
-- ============================================================

-- Returns (value, next_pos) or errors.
local decode_value  -- forward declaration

local function skip_whitespace(s, pos)
    return s:match('^%s*()', pos)
end

local function decode_string(s, pos)
    -- pos points at the opening "
    local result = {}
    local i = pos + 1  -- skip opening quote
    while i <= #s do
        local c = s:sub(i, i)
        if c == '"' then
            return table.concat(result), i + 1
        elseif c == '\\' then
            i = i + 1
            local esc = s:sub(i, i)
            if esc == '"'  then result[#result+1] = '"'
            elseif esc == '\\' then result[#result+1] = '\\'
            elseif esc == '/' then result[#result+1] = '/'
            elseif esc == 'n'  then result[#result+1] = '\n'
            elseif esc == 'r'  then result[#result+1] = '\r'
            elseif esc == 't'  then result[#result+1] = '\t'
            elseif esc == 'b'  then result[#result+1] = '\b'
            elseif esc == 'f'  then result[#result+1] = '\f'
            elseif esc == 'u'  then
                local hex = s:sub(i+1, i+4)
                local codepoint = tonumber(hex, 16)
                if not codepoint then
                    error('json.decode: invalid \\u escape at pos ' .. i)
                end
                -- Encode codepoint as UTF-8
                if codepoint < 0x80 then
                    result[#result+1] = string.char(codepoint)
                elseif codepoint < 0x800 then
                    result[#result+1] = string.char(
                        0xC0 + math.floor(codepoint / 64),
                        0x80 + (codepoint % 64))
                else
                    result[#result+1] = string.char(
                        0xE0 + math.floor(codepoint / 4096),
                        0x80 + math.floor((codepoint % 4096) / 64),
                        0x80 + (codepoint % 64))
                end
                i = i + 4
            else
                error('json.decode: unknown escape \\' .. esc .. ' at pos ' .. i)
            end
        else
            result[#result+1] = c
        end
        i = i + 1
    end
    error('json.decode: unterminated string starting at pos ' .. pos)
end

local function decode_number(s, pos)
    local num_str = s:match('^-?%d+%.?%d*[eE]?[+-]?%d*', pos)
    if not num_str then
        error('json.decode: invalid number at pos ' .. pos)
    end
    return tonumber(num_str), pos + #num_str
end

local function decode_array(s, pos)
    -- pos points past the opening [
    local arr = {}
    local i = skip_whitespace(s, pos)
    if s:sub(i, i) == ']' then
        return arr, i + 1
    end
    while true do
        local val, next_i = decode_value(s, i)
        arr[#arr + 1] = val
        i = skip_whitespace(s, next_i)
        local c = s:sub(i, i)
        if c == ']' then
            return arr, i + 1
        elseif c == ',' then
            i = skip_whitespace(s, i + 1)
        else
            error('json.decode: expected , or ] in array at pos ' .. i)
        end
    end
end

local function decode_object(s, pos)
    -- pos points past the opening {
    local obj = {}
    local i = skip_whitespace(s, pos)
    if s:sub(i, i) == '}' then
        return obj, i + 1
    end
    while true do
        i = skip_whitespace(s, i)
        if s:sub(i, i) ~= '"' then
            error('json.decode: expected string key at pos ' .. i)
        end
        local key, key_end = decode_string(s, i)
        i = skip_whitespace(s, key_end)
        if s:sub(i, i) ~= ':' then
            error('json.decode: expected : after key at pos ' .. i)
        end
        i = skip_whitespace(s, i + 1)
        local val, val_end = decode_value(s, i)
        obj[key] = val
        i = skip_whitespace(s, val_end)
        local c = s:sub(i, i)
        if c == '}' then
            return obj, i + 1
        elseif c == ',' then
            i = skip_whitespace(s, i + 1)
        else
            error('json.decode: expected , or } in object at pos ' .. i)
        end
    end
end

decode_value = function(s, pos)
    pos = skip_whitespace(s, pos)
    local c = s:sub(pos, pos)
    if c == '"' then
        return decode_string(s, pos)
    elseif c == '{' then
        return decode_object(s, pos + 1)
    elseif c == '[' then
        return decode_array(s, pos + 1)
    elseif c == 't' then
        if s:sub(pos, pos+3) == 'true' then return true, pos + 4 end
        error('json.decode: invalid token at pos ' .. pos)
    elseif c == 'f' then
        if s:sub(pos, pos+4) == 'false' then return false, pos + 5 end
        error('json.decode: invalid token at pos ' .. pos)
    elseif c == 'n' then
        if s:sub(pos, pos+3) == 'null' then return nil, pos + 4 end
        error('json.decode: invalid token at pos ' .. pos)
    elseif c == '-' or c:match('%d') then
        return decode_number(s, pos)
    else
        error('json.decode: unexpected character "' .. c .. '" at pos ' .. pos)
    end
end

function json.decode(s)
    if type(s) ~= 'string' then
        error('json.decode: expected string, got ' .. type(s))
    end
    local val, pos = decode_value(s, 1)
    pos = skip_whitespace(s, pos)
    if pos <= #s then
        error('json.decode: trailing garbage at pos ' .. pos)
    end
    return val
end

-- Sentinel for null (distinct from Lua nil)
json.null = setmetatable({}, { __tostring = function() return 'null' end })

return json
