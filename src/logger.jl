# Log verbosity level
global LOG_LEVEL = 0

function loglevel!(ll::Int)
    global LOG_LEVEL
    LOG_LEVEL = ll
end

# RED (31): Error (1)
# BLUE (34): Info (2)
# DEBUG (32): Debug (3)
# DEV (36): Development (4)
# Trace (33): Trace (5)
function LOG(lvl, msg...)
    if LOG_LEVEL < lvl
        return
    else
        #prefix = lvl == 1 ? "\e[1;34mINFO" : (lvl == 2 ? "\e[1;32mDEBUG" : "\e[1;33mTRACE")
        #println(prefix, ": ", msg..., "\e[0m")
        prefix = lvl == 1 ? "\e[1;31m" : (lvl == 2 ? "\e[1;34m" :
            (lvl == 3 ? "\e[1;32m" : (lvl == 4 ? "\e[1;36m" : "\e[1;33m")))
        println(prefix, msg..., "\e[0m")
    end
end
