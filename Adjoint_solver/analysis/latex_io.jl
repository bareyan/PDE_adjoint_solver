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
minimum run time and allocation count from a single time-budgeted trial.

`evals = 1` skips BenchmarkTools' auto-tuning, which would otherwise call `f`
several times just to calibrate — fatal for the naive AD methods, where one
evaluation can take minutes. Time and memory are read off the *same* trial, so a
slow method costs about one warm-up plus one timed evaluation.
"""
function benchmark(f; seconds = 5)
    value = f()                                      # warm up (compile) + value
    trial = @benchmark $f() samples = 100 evals = 1 seconds = seconds
    return (value = value,
            time_s = minimum(trial).time / 1e9,
            alloc_bytes = minimum(trial).memory)
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
                     column_labels = [names(df)],   # header only — drop the type-annotation subrow
                     kwargs...)
    end
end

"""
    write_tables_by(datadir, name, df, by; columns = Not(by), caption = "", label = "tab:"*name)

Write one booktabs LaTeX table per group of `df` (grouped on column `by`) into a
single <datadir>/<name>.tex. Each group becomes its own `table`, captioned with the
group value and labelled `<label>-<by><value>`. The `[H]` placement (float package)
pins the tables in source order, so a long run of them stays put where included
(e.g. in an appendix) instead of floating ahead of the surrounding text.
"""
function write_tables_by(datadir, name, df, by::Symbol;
                         columns = Not(by), caption = "", label = "tab:" * name)
    mkpath(datadir)
    open(joinpath(datadir, "$name.tex"), "w") do io
        for sub in groupby(df, by)
            key = sub[1, by]
            println(io, "\\begin{table}[H]")
            println(io, "  \\centering")
            println(io, "  \\caption{$caption (\$$by = $key\$).}")
            println(io, "  \\label{$label-$by$key}")
            sel = select(sub, columns)
            pretty_table(io, sel; backend = :latex,
                         table_format = PrettyTables.latex_table_format__booktabs,
                         column_labels = [names(sel)])   # header only — drop the type-annotation subrow
            println(io, "\\end{table}")
        end
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
    git = try readchomp(`git rev-parse --short HEAD`) catch; "n/a" end
    cpu = try Sys.cpu_info()[1].model catch; "unknown CPU" end
    # `pkgversion` on the loaded module finds versions regardless of which stacked
    # environment provides the package (Pkg.dependencies only sees the active project).
    open(joinpath(datadir, "meta.tex"), "w") do io
        print(io, cpu, " (", Sys.CPU_THREADS, " threads); Julia ", VERSION,
              ", BLAS threads ", BLAS.get_num_threads())
        for p in packages
            m = try getfield(Main, Symbol(p)) catch; nothing end
            m isa Module && print(io, ", $p ", pkgversion(m))
        end
        print(io, "; commit ", git, ", ", Dates.format(now(), "yyyy-mm-dd"), ".")
    end
end
