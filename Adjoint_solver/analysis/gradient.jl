# Compares gradient methods for the inverse problem (-u'' + αu³ + Vu = f):
# finite differences, naive AD (forward/reverse), the hand adjoint, and rule-based
# AD. Writes tables/figures/meta into gradient_data/ ; gradient.tex includes them.
#
# Two state solves appear here. 
# The production Newton_solve (Krylov.cg + FFT) is used for FD and as the primal of the adjoint and rule-based paths. 
# The naive AD paths use newton_ad: a Newton loop over spectral operators,
# so the iteration graph is AD-traceable. Its inner linear solve calls
# the matrix-form cg_ad, which carries a custom rule (gradient/rules.jl), so
# reverse-mode AD differentiates only the Newton iteration, never CG.

#### Setup
const SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(SRC, "solvers.jl"))             # Problem, Newton_solve, F, cg_ad, FD, *_gradient
include(joinpath(SRC, "gradient", "rules.jl"))   # cg_ad rule, Newton_solve adjoint rule
include(joinpath(@__DIR__, "latex_io.jl"))       # results_frame, measure!, write_*

using ForwardDiff, Mooncake, Zygote, Plots, LinearAlgebra

const DATA         = joinpath(@__DIR__, "gradient_data")
const NS           = [5, 10, 25, 50, 100, 500]
const NEWTON_STEPS = 100

make_problem(N) = Problem(N = N, L = 2π, α = 1.0, f = x -> 2 + cos(2x) + sin(x))

#### Precalculated spectral operators for the naive AD, not to differentiate fft and ifft.(causes problems)
function spectral_operators(p; c = 2.0)
    D2   = zeros(p.N, p.N)
    Minv = zeros(p.N, p.N)
    for j in 1:p.N
        e = zeros(p.N); e[j] = 1.0
        D2[:, j]   = diff2_fourier(e, p.ks)
        Minv[:, j] = real(ifft(fft(e) ./ (p.ks .^ 2 .+ c)))
    end
    return D2, Minv
end

#### AD-friendly state solve: uses the spectral operators, and differentiable cg_ad.
function newton_ad(p, V, D2, Minv; n_iter = NEWTON_STEPS)
    u = zeros(eltype(V), p.N)
    for _ in 1:n_iter
        res = -D2 * u .+ p.α .* u .^ 3 .+ V .* u .- p.f
        J   = -D2 + Diagonal(3 .* p.α .* u .^ 2 .+ V)
        u   = u .- cg_ad(J, res; M = Minv)        # matrix-form cg_ad → rule fires
        if l2norm(res) < 1e-16
            break
        end
    end
    return u
end

#### Losses  L(V) = (L/N) Σ (u(V) - t)², one per state solve.
loss_real(p, V, t)         = (p.L / p.N) * sum((Newton_solve(p, V) .- t) .^ 2)
loss_ad(p, V, t, D2, Minv) = (p.L / p.N) * sum((newton_ad(p, V, D2, Minv) .- t) .^ 2)

#### Gradient methods, uniform signature g(p, V, ut, ops) -> grad. ops = (D2, Minv).
grad_fd(p, V, ut, ops)      = FD(p, (v, t) -> loss_real(p, v, t), V, ut)
grad_adjoint(p, V, ut, ops) = adjoint_gradient(p, V, ut)[2]

grad_naive_forward(p, V, ut, ops)  = forwarddiff_gradient(v -> loss_ad(p, v, ut, ops...), V)
grad_naive_mooncake(p, V, ut, ops) = mooncake_gradient(v -> loss_ad(p, v, ut, ops...), V)
grad_naive_zygote(p, V, ut, ops)   = zygote_gradient(v -> loss_ad(p, v, ut, ops...), V)

grad_ruled_mooncake(p, V, ut, ops) = mooncake_gradient(v -> loss_real(p, v, ut), V)
grad_ruled_zygote(p, V, ut, ops)   = zygote_gradient(v -> loss_real(p, v, ut), V)

#                 category    tool           mode        gradient(p, V, ut, ops)
const METHODS = [("FD",      "-",           "-",       grad_fd),
                 ("adjoint", "-",           "-",       grad_adjoint),  # analytic; accuracy reference
                 ("naive",   "ForwardDiff", "forward", grad_naive_forward),
                 ("naive",   "Mooncake",    "reverse", grad_naive_mooncake),
                 ("naive",   "Zygote",      "reverse", grad_naive_zygote),
                 ("ruled",   "Mooncake",    "reverse", grad_ruled_mooncake),
                 ("ruled",   "Zygote",      "reverse", grad_ruled_zygote)]

function main(; ns = NS, datadir = DATA, seconds = 5)
    grid_sizes = collect(ns)
    isempty(grid_sizes) && throw(ArgumentError("ns must contain at least one grid size"))
    df = results_frame()
    for N in grid_sizes
        p   = make_problem(N)
        ut  = Newton_solve(p, 1 .+ 0.5 .* (p.x .- π) .^ 2)
        V0  = 1.5 .* ones(N)
        ref = adjoint_gradient(p, V0, ut)[2]          # accuracy reference
        ops = spectral_operators(p)                   # dense D2, Minv, outside timing
        for (category, tool, mode, g) in METHODS
            measure!(df, () -> g(p, V0, ut, ops);
                     category = category, tool = tool, mode = mode,
                     N = N, ref = ref, seconds = seconds)
        end
        println("N = $N done")
    end

    final_N = last(grid_sizes)
    write_table(datadir, "cost_table",     filter(:N => ==(final_N), df))
    write_tables_by(datadir, "scaling_table", df, :N; columns = Not(:N),
                    caption = "Time, allocations and relative \$\\ell_2\$ error against the analytic adjoint",
                    label = "tab:scaling")
    write_table(datadir, "accuracy_table",
                select(filter(:N => ==(final_N), df), [:category, :tool, :mode, :rel_err]))
    save_figure(datadir, "scaling_time",        scaling_plot(df, :time_s,      "time (s)"))
    save_figure(datadir, "scaling_memory",      scaling_plot(df, :alloc_bytes, "alloc (bytes)"))
    save_figure(datadir, "accuracy_vs_adjoint", accuracy_plot(filter(:N => ==(final_N), df)))
    write_meta(datadir; packages = ["ForwardDiff", "Mooncake", "Zygote"])
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

function accuracy_plot(df)
    sub = filter(:category => !=("adjoint"), df)   # drop the zero reference (no log of 0)
    bar(sub.category .* " " .* sub.tool, sub.rel_err;
        yscale = :log10, ylabel = "rel. l² error vs adjoint",
        legend = false, xrotation = 45)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
