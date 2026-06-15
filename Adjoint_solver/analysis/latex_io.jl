# analysis/latex_io.jl
#
# Shared helpers for the analysis scripts (gradient_computation.jl, hessian.jl,
# optimizers.jl, regularization.jl): benchmarking, and writing LaTeX-ready
# tables / figures / provenance. Each analysis writes into its own <topic>_data/
# folder; the matching <topic>.tex \input's the tables and \includegraphics the PDFs.

using DataFrames, PrettyTables, BenchmarkTools, LinearAlgebra, Dates
import Pkg
import LinearAlgebra.BLAS

# --- benchmarking ----------------------------------------------------------

const DEFAULT_RESULT_COLUMNS = (:category => String, :tool => String, :mode => String, :N => Int)

_empty_column(::Type{T}) where {T} = T[]
_empty_column(v::AbstractVector) = copy(v)
_empty_column(v) = typeof(v)[]

function _columns_from_schema(schema)
    names = Tuple(Symbol(first(pair)) for pair in schema)
    values = Tuple(_empty_column(last(pair)) for pair in schema)
    return NamedTuple{names}(values)
end

"""
    benchmark(f; seconds = 5) -> (value, time_s, alloc_bytes)

Warm up `f` (a zero-argument closure), then return its value together with its
minimum run time and allocation count. The universal building block for every
timing study.
"""
function benchmark(f; seconds = 5)
    v = f()                                          # warm up + value
    t = @belapsed $f() seconds = seconds
    a = @ballocated $f()
    return (value = v, time_s = t, alloc_bytes = a)
end

"""
    relative_error(value, ref) -> Float64

Relative L2 error for scalar or array-like benchmark results. If `ref` has zero
norm, this returns the absolute L2 error instead.
"""
function relative_error(value, ref)
    numerator = norm(value .- ref)
    denominator = norm(ref)
    return denominator == 0 ? numerator : numerator / denominator
end

"""
    measure!(df, f; ref = nothing, seconds = 5, metadata...) -> value

Benchmark `f` and push one row into `df`: the `metadata` columns plus `time_s`,
`alloc_bytes`, and, when `df` has a `rel_err` column, the relative deviation of
the result from `ref` (`NaN` if no reference). Returns `f`'s value.
"""
function measure!(df, f; ref = nothing, seconds = 5, metadata...)
    m = benchmark(f; seconds)
    row = if :rel_err in propertynames(df)
        e = ref === nothing ? NaN : relative_error(m.value, ref)
        (; metadata..., time_s = m.time_s, alloc_bytes = m.alloc_bytes, rel_err = e)
    else
        (; metadata..., time_s = m.time_s, alloc_bytes = m.alloc_bytes)
    end
    push!(df, row)
    return m.value
end

"""
    results_frame(:method => String, :N => Int; include_rel_err = true) -> DataFrame

Empty results table: the given metadata columns plus the measured `time_s`,
`alloc_bytes`, and optionally `rel_err`. With no schema, defaults to
(`category, tool, mode, N`).
"""
function results_frame(schema::Pair...; include_rel_err = true)
    cols = _columns_from_schema(isempty(schema) ? DEFAULT_RESULT_COLUMNS : schema)
    df = DataFrame(; cols..., time_s = Float64[], alloc_bytes = Int[])
    include_rel_err && (df.rel_err = Float64[])
    return df
end

# --- LaTeX artifacts -------------------------------------------------------

"Write `df` as a booktabs LaTeX table into <datadir>/<name>.tex."
function write_table(datadir, name, df; kwargs...)
    mkpath(datadir)
    open(joinpath(datadir, "$name.tex"), "w") do io
        pretty_table(io, df; backend = :latex,
                     table_format = PrettyTables.latex_table_format__booktabs,
                     kwargs...)
    end
end

"Save plot `plt` as <datadir>/<name>.pdf (the caller must have Plots loaded)."
function save_figure(datadir, name, plt)
    mkpath(datadir)
    savefig(plt, joinpath(datadir, "$name.pdf"))
end

"Stamp environment provenance into <datadir>/meta.tex for \\input in the report."
function write_meta(datadir; packages = String[])
    mkpath(datadir)
    vers = Dict(i.name => i.version for (_, i) in Pkg.dependencies() if i.name in packages)
    git  = try readchomp(`git rev-parse --short HEAD`) catch; "n/a" end
    open(joinpath(datadir, "meta.tex"), "w") do io
        print(io, "Julia ", VERSION, ", BLAS threads ", BLAS.get_num_threads())
        for p in packages
            haskey(vers, p) && print(io, ", $p ", vers[p])
        end
        print(io, "; commit ", git, ", ", Dates.format(now(), "yyyy-mm-dd"), ".")
    end
end
