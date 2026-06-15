# analysis/gradient_computation.jl
#
# Compares gradient methods for the inverse problem (-u'' + αu³ + Vu = f):
# finite differences, naive AD (forward/reverse), the hand adjoint, rule-based AD.
# Writes tables/figures/meta into gradient_data/ ; gradient.tex includes them.
#
# AD note: the production Newton_solve (Krylov.cg + FFT) is not AD-traceable, so
# the naive AD paths use Newton_solve_ad (plain cg_ad + diff matrices)

#### Setup
const SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(SRC, "solvers.jl"))             # Problem, Newton_solve, F, cg_ad, ...
include(joinpath(SRC, "gradient", "adjoint.jl")) # adjoint_gradient
include(joinpath(SRC, "gradient", "simple.jl"))  # FD, Forward_AD, Backward_AD
include(joinpath(SRC, "gradient", "rules.jl")) 
include(joinpath(@__DIR__, "latex_io.jl"))       # results_frame, measure!, write_*

using Enzyme, Mooncake, Zygote, Plots

Enzyme.@import_frule typeof(cg_ad) AbstractMatrix AbstractVector
Enzyme.@import_rrule typeof(cg_ad) AbstractMatrix AbstractVector

Enzyme.API.strictAliasing!(false)

const DATA = joinpath(@__DIR__, "gradient_data")
# const NS   = [5, 10, 25, 50, 100, 200, 350, 500]
const NS   = [5, 50, 100,  500]
const NEWTON_STEPS = 100

make_problem(N) = Problem(N = N, L = 2π, α = 1.0, f = x -> 2 + cos(2x) + sin(x))


D2 = nothing
Minv = nothing


#### AD-friendly state solve
function naive_operators(p; c = 2.0)
    global D2, Minv
    D2   = zeros(p.N, p.N)
    Minv = zeros(p.N, p.N)
    for j in 1:p.N
        e = zeros(p.N); e[j] = 1.0
        D2[:, j]   = real(ifft(-p.ks.^2 .* fft(e)))
        Minv[:, j] = real(ifft(fft(e) ./ (p.ks.^2 .+ c)))
    end
    return nothing
end



function Newton_solve_ad(p, V; n_iter = NEWTON_STEPS)
    u = zeros(eltype(V), p.N)
    for _ in 1:n_iter                      # fixed count → static graph for AD
        res       = -D2 * u .+ p.α .* u.^3 .+ V .* u .- p.f
        applyJ    = δu -> -D2 * δu .+ (3 * p.α .* u.^2 .+ V) .* δu
        applyMinv = z  -> Minv * z
        u = u - cg_ad(applyJ, res; M = applyMinv)
    end
    return u
end


#### Losses  (two-arg loss(V, target), as simple.jl's FD/AD expect)
loss_real(p, V, t)         = (p.L / p.N) * sum((Newton_solve(p, V) .- t).^2)
loss_ad(p, V, t; n_iter = NEWTON_STEPS) =
    (p.L / p.N) * sum((Newton_solve_ad(p, V; n_iter = n_iter) .- t).^2)

#### Gradient methods, each as ctx -> grad.  ctx = (; p, ut, V, D2, Minv).
# The AD methods build their loss closure fresh on every call (the fix for the
# global-loss bug); D2/Minv come from ctx, built once per N outside the timing.

function FD_problem(p, V, ut)
    return FD(p, (V,t)->loss_real(p, V, t), V, ut)
end

function Adjoint_problem(p, V, ut)
    return adjoint_gradient(p, V, ut)[2]
end

function Forward_AD_problem(p, V, ut; n_iter = NEWTON_STEPS)
    loss = v -> loss_ad(p, v, ut; n_iter = n_iter)
    return Forward_AD(loss, V)
end

function Backward_AD_problem(p, V, ut; n_iter = NEWTON_STEPS)
    loss = v -> loss_ad(p, v, ut; n_iter = n_iter)
    return Backward_AD(loss, V)
end

#                 category    tool        mode       gradient(ctx)
const METHODS = [("FD",      "-",        "-",       FD_problem),
                 ("adjoint", "-",        "reverse",       Adjoint_problem),
                #  ("ruled",   "Enzyme",   "reverse", c -> ...),   # TODO Newton rule
                #  ("ruled",   "Mooncake", "reverse", c -> ...),
                #  ("ruled",   "Zygote",   "reverse", c -> ...),
                 ]


function main(; ns = NS, datadir = DATA, seconds = 5)
    grid_sizes = collect(ns)
    isempty(grid_sizes) && throw(ArgumentError("ns must contain at least one grid size"))
    df = results_frame()
    for N in grid_sizes
        p  = make_problem(N)
        ut = Newton_solve(p, 1 .+ 0.5 .* (p.x .- π).^2)
        V0  = 1.5 .* ones(N)
        ref = adjoint_gradient(p, V0, ut)[2] # accuracy reference
        
        # for (cat, tool, mode, g) in METHODS
        #     measure!(df, () -> g(p, V0, ut);
        #              category = cat, tool = tool, mode = mode, N = N,
        #              ref = ref, seconds = seconds)
        # end

        ## Naive AD methods (forward/reverse) need D2/Minv built once per N
        naive_operators(p; c = 2.0)
        println("N=$N, D2/Minv built", " (size(D2)=$(size(D2)), size(Minv)=$(size(Minv)))")

        function Forward_AD_problem(p, V, ut; n_iter = NEWTON_STEPS)
            loss = v -> loss_ad(p, v, ut; n_iter = n_iter)
            return Forward_AD(loss, V)
        end

        function Backward_AD_problem(p, V, ut; n_iter = NEWTON_STEPS)
            loss = v -> loss_ad(p, v, ut; n_iter = n_iter)
            return Backward_AD(loss, V)
        end

        measure!(df, () -> Forward_AD(v -> loss_ad(p, v, ut; n_iter = 100), V0);
                 category = "Forward_AD", tool = "Enzyme", mode = "forward",
                 N = N, ref = ref, seconds = seconds)

        measure!(df, () -> Backward_AD(v -> loss_ad(p, v, ut; n_iter = 100), V0);
                 category = "Backward_AD", tool = "Enzyme", mode = "reverse",
                 N = N, ref = ref, seconds = seconds)

        println("N=$N done")
    end

    final_N = last(grid_sizes)
    write_table(datadir, "cost_table",     filter(:N => ==(final_N), df))
    write_table(datadir, "scaling_table",  select(df, Not(:rel_err)))
    write_table(datadir, "accuracy_table", filter(:N => ==(final_N),
                                               select(df, [:category, :tool, :mode, :rel_err])))
    save_figure(datadir, "scaling_time",        scaling_plot(df, :time_s,      "time (s)"))
    save_figure(datadir, "scaling_memory",      scaling_plot(df, :alloc_bytes, "alloc (bytes)"))
    save_figure(datadir, "accuracy_vs_adjoint", accuracy_plot(filter(:N => ==(final_N), df)))
    write_meta(datadir; packages = ["Enzyme", "Mooncake", "Zygote"])
    return df
end

#### Study-specific plots
function scaling_plot(df, col, ylab)
    plt = plot(xscale = :log10, yscale = :log10, xlabel = "N", ylabel = ylab, legend = :topleft)
    for sub in groupby(df, [:category, :tool, :mode])
        label = join(filter(!=("-"), [sub.category[1], sub.tool[1], sub.mode[1]]), " ")
        plot!(plt, sub.N, sub[!, col]; marker = :circle, label)
    end
    plt
end

accuracy_plot(df) = bar(df.category .* " " .* df.tool, df.rel_err;
                        yscale = :log10, ylabel = "rel. ℓ² error vs adjoint",
                        legend = false, xrotation = 45)

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
