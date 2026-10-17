# Output helpers: CSV and JSON writers and a minimal command-line parser.

"""
    rows_to_dataframe(rows)

DataFrame from a vector of NamedTuples that share the same field names, with
the columns in field order. NaN becomes `missing`, which CSV.jl writes as an
empty field, as pandas does with NaN.
"""
function rows_to_dataframe(rows::AbstractVector)
    isempty(rows) && return DataFrame()
    df = DataFrame()
    for name in keys(first(rows))
        df[!, name] = [nan_to_missing(getproperty(r, name)) for r in rows]
    end
    return df
end

nan_to_missing(x::AbstractFloat) = isnan(x) ? missing : x
nan_to_missing(x) = x

"""
    write_rows_csv(path, rows)

Write NamedTuple rows to a CSV file with the same column order as the Python
output.
"""
function write_rows_csv(path::AbstractString, rows::AbstractVector)
    CSV.write(path, rows_to_dataframe(rows))
    return path
end

# --- JSON ------------------------------------------------------------------
# Objects are given as vectors of `String => value` pairs so that key order is
# preserved. Nested vectors and objects are supported. Build objects with
# `jobj` so that integer values are not promoted to floating point.

const JSONObject = AbstractVector{<:Pair}

"""jobj(pairs...) builds an ordered JSON object as a Vector{Pair{String,Any}}."""
jobj(pairs::Pair...) = Pair{String,Any}[string(k) => v for (k, v) in pairs]

function json_escape(s::AbstractString)
    io = IOBuffer()
    for ch in s
        if ch == '"'
            print(io, "\\\"")
        elseif ch == '\\'
            print(io, "\\\\")
        elseif ch == '\n'
            print(io, "\\n")
        elseif ch == '\t'
            print(io, "\\t")
        elseif ch < ' '
            print(io, "\\u", string(UInt16(ch), base = 16, pad = 4))
        else
            print(io, ch)
        end
    end
    return String(take!(io))
end

json_write(io::IO, x::AbstractString, indent::Int) = print(io, '"', json_escape(x), '"')
json_write(io::IO, x::Symbol, indent::Int) = json_write(io, String(x), indent)
json_write(io::IO, x::Bool, indent::Int) = print(io, x ? "true" : "false")
json_write(io::IO, x::Integer, indent::Int) = print(io, x)
json_write(io::IO, x::Nothing, indent::Int) = print(io, "null")
function json_write(io::IO, x::AbstractFloat, indent::Int)
    isfinite(x) ? print(io, repr(Float64(x))) : print(io, "null")
end
function json_write(io::IO, x::Union{AbstractVector,Tuple}, indent::Int)
    if x isa JSONObject
        return json_write_object(io, x, indent)
    end
    if isempty(x)
        print(io, "[]")
        return
    end
    pad = " "^(indent + 2)
    println(io, "[")
    for (i, v) in enumerate(x)
        print(io, pad)
        json_write(io, v, indent + 2)
        println(io, i < length(x) ? "," : "")
    end
    print(io, " "^indent, "]")
end
function json_write_object(io::IO, x, indent::Int)
    if isempty(x)
        print(io, "{}")
        return
    end
    pad = " "^(indent + 2)
    println(io, "{")
    for (i, (k, v)) in enumerate(x)
        print(io, pad, '"', json_escape(string(k)), "\": ")
        json_write(io, v, indent + 2)
        println(io, i < length(x) ? "," : "")
    end
    print(io, " "^indent, "}")
end

"""
    write_json(path, pairs)

Write an ordered JSON object given as a vector of `key => value` pairs, with
two-space indentation as in Python's json.dumps(indent=2).
"""
function write_json(path::AbstractString, pairs::AbstractVector{<:Pair})
    open(path, "w") do io
        json_write_object(io, pairs, 0)
        println(io)
    end
    return path
end

# --- Command line ------------------------------------------------------------

"""
    parse_cli(args, defaults) -> Dict{String,Any}

Parse `--name value` options against a dictionary of defaults. The type of each
default decides how the value is read: Int, Float64, String, a vector of
Float64 given as comma-separated numbers (for example `--nus 3,5,10`), or Bool
for a switch that takes no value (for example `--grid-check-only`).
"""
function parse_cli(args::AbstractVector{<:AbstractString}, defaults::AbstractDict)
    opts = Dict{String,Any}(defaults)
    i = 1
    while i <= length(args)
        arg = args[i]
        startswith(arg, "--") || error("unexpected argument '$arg'")
        name = arg[3:end]
        if !haskey(defaults, name)
            known = join(sort!(collect(keys(defaults))), ", ")
            error("unknown option --$name (known options: $known)")
        end
        default = defaults[name]
        if default isa Bool
            opts[name] = true
            i += 1
        else
            i < length(args) || error("option --$name needs a value")
            opts[name] = parse_option(default, args[i + 1])
            i += 2
        end
    end
    return opts
end

parse_option(::Integer, v::AbstractString) = parse(Int, v)
parse_option(::AbstractFloat, v::AbstractString) = parse(Float64, v)
parse_option(::AbstractString, v::AbstractString) = String(v)
parse_option(::AbstractVector{<:Real}, v::AbstractString) =
    [parse(Float64, strip(p)) for p in split(v, ',') if !isempty(strip(p))]

"""Short description of the Julia session, stored in the metadata files."""
function session_info()
    return jobj(
        "julia" => string(VERSION),
        "platform" => string(Sys.MACHINE),
        "threads" => Threads.nthreads(),
        "blas_threads" => LinearAlgebra.BLAS.get_num_threads(),
        "rng" => "Random.Xoshiro; one stream per design cell, seeded from the master seed",
    )
end
