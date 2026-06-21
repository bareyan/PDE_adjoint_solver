#########################################################################################
# Experiments for the inverse problem of recovering the potential V from the state u(V) of the PDE -u'' + αu³ + Vu = f.
# Compares behavior for well-conditioned vs ill-conditioned problems
# Analyses the hessian spectrum, optimizer performance and regularization effects.
# Tests are for 
# - Different optimizers (GD, Adam, L-BFGS)
# - Different nonlinearity strengths (α)
# - Different regularization methods (R1, R2)
# Exports plots and tables to optimizers_data/
#########################################################################################


#### Setup
const SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(SRC, "Adjoint_solver.jl"))
using .Adjoint_solver
include(joinpath(@__DIR__, "latex_io.jl"))


using LinearAlgebra, FFTW, Random
using DataFrames, Plots

const DATA   = joinpath(@__DIR__, "optimizers_data"); mkpath(DATA)

#### Hyper Parameters
const N      = 500
const α₀     = 10.0
const TOL    = 1e-12      # "converged" loss
const TARGET = 1e-6       # loss level for the time-to-target benchmark

#### Problem setup
u_well(x) = 2 + sin(x)          # strictly positive => well-conditioned / identifiable
u_ill(x)  = sin(x)              # zero-mean, sign-changing => diag(u) singular => ill-conditioned
Vtrue(x)  = 3 + cos(x)
V_spike(x) = 2.0 + 3.0 * (abs(x - π / 2) < 0.4) 

function instance(u_fn, α; n = N, V_fn = Vtrue)
    p0 = Problem(N = n, L = 2π, α = α, f = zeros(n))
    x  = p0.x
    u  = u_fn.(x)
    u ./= sqrt((p0.L / p0.N) * sum(abs2, u)) 
    V  = V_fn.(x)
    f  = -diff2_fourier(u, p0.ks) .+ α .* u .^ 3 .+ V .* u
    (; p = Problem(N = n, L = 2π, α = α, f = f), u_target = u, Vtrue = V, α)
end

# Same instance with relative white noise added to the data u_target (same recipe as the β-sweep).
function noisy(inst; noise = 1e-4, seed = 1)
    u_n = inst.u_target .+ (noise * norm(inst.u_target) / √N) .* randn(MersenneTwister(seed), N)
    (; inst..., u_target = u_n)
end

function setup_check()
    ## Plot the well and ill conditioned instances for a reference in the tex
    inst_well = instance(u_well, α₀)
    inst_ill  = instance(u_ill, α₀)
    i_well = plot(plot(inst_well.p.x, inst_well.u_target; label = "u_target", title = "well-conditioned instance", xlabel = "x", ylabel = "u", ylims = (-1, 1)),
                  plot(inst_well.p.x, inst_well.Vtrue; label = "V_true", title = "well-conditioned instance", xlabel = "x", ylabel = "V", ylims = (0, 5)),
                  layout = (1, 2), size = (900, 400))
    save_figure(DATA, "instance_well", i_well)
    i_ill = plot(plot(inst_ill.p.x, inst_ill.u_target; label = "u_target", title = "ill-conditioned instance", xlabel = "x", ylabel = "u", ylims = (-1, 1)),
                 plot(inst_ill.p.x, inst_ill.Vtrue; label = "V_true", title = "ill-conditioned instance", xlabel = "x", ylabel = "V", ylims = (0, 5)),
                 layout = (1, 2), size = (900, 400))
    save_figure(DATA, "instance_ill", i_ill)
end


loss_of(inst, V; reg = NoReg()) = loss(V; p = inst.p, u_target = inst.u_target, reg = reg)
grad_of(inst, V; reg = NoReg()) = zygote_gradient(V -> loss(V; p = inst.p, u_target = inst.u_target, reg = reg), V)
 #adjoint_gradient(inst.p, V, inst.u_target; reg = reg)[2]
#
pot_err(inst, V)                = norm(V .- inst.Vtrue) / norm(inst.Vtrue)

optimizer(name, budget) = name == "gd"   ? OGradientDescent(n_iter = budget, η = 1e-2, gtol = 1e-12) :
                          name == "adam" ? OAdam(n_iter = budget, η = 1e-2, gtol = 1e-12) :
                                           OLBFGS(n_iter = budget, m = 15, gtol = 1e-12)

# One run. `timed` adds wall time via the latex_io benchmark helper.
# Metric records (loss, pot_err) per iteration ⇒ `hist` is the loss curve, `pe_hist` the V-error curve.
# `record = false` skips that logging (each entry is a PDE solve) when only the final V is needed — e.g. the β sweep.
function runopt(inst, o; reg = NoReg(), timed = false, record = true)

    metric = record ? (V -> (loss_of(inst, V; reg), pot_err(inst, V))) : (V -> (0.0, 0.0))
    work() = optimize(o, V -> loss_of(inst, V; reg), V -> grad_of(inst, V; reg), ones(N); history_step = 1, metric)
    V, h, t = timed ? (m = benchmark(work); (m.value..., m.time_s)) : (work()..., NaN)

    hist, pe_hist = first.(h), last.(h)
    itol = record ? findfirst(<(TOL), hist) : nothing
    (; V, hist, pe_hist, time_s = t, iters = length(hist), final_loss = loss_of(inst, V; reg),
       pot_err = pot_err(inst, V), iters_to_tol = itol === nothing ? -1 : itol)
end

const WELL = instance(u_well, α₀)
const ILL  = instance(u_ill,  α₀)

# ============================================================================
# Experiment 1 — Optimizer comparison.
#   Which optimizer recovers V fastest?  gd vs adam vs lbfgs, on well and ill.
#   Export: loss-vs-iteration plot (one per instance) + table (time/iters to target & tol, recovery).
# ============================================================================
function experiment_optimizers()
    df = DataFrame(config = String[], optimizer = String[], it_to_target = Int[], s_to_target = Float64[],
                   it_to_tol = Int[], final_loss = Float64[], pot_err = Float64[], total_s = Float64[])
    for (name, inst) in (("well", WELL), ("ill", ILL))
        plt1 = plot(yscale = :log10, xscale = :log10, xlabel = "iteration", ylabel = "loss",
                    title = "loss vs iteration — $name", legend = :topright)
        plt2 = plot(yscale = :log10, xscale = :log10, xlabel = "iteration", ylabel = "pot_err",
                    title = "V error vs iteration — $name", legend = :topright)
        for oname in ("gd", "adam", "lbfgs")
            r = runopt(inst, optimizer(oname, 3000); timed = true)
            it_t = findfirst(<(TARGET), r.hist)
            push!(df, (name, oname, something(it_t, -1), it_t === nothing ? NaN : it_t * r.time_s / r.iters,
                       r.iters_to_tol, r.final_loss, r.pot_err, r.time_s))
            plot!(plt1, max.(r.hist, 1e-20); label = oname)
            plot!(plt2, max.(r.pe_hist, 1e-20); label = oname)
        end
        plt = plot(plt1, plt2; layout = (1, 2), size = (900, 400))
        save_figure(DATA, "optimizers_loss_$name", plt)
    end
    write_table(DATA, "optimizers", df)
    df
end

# Exact Hessian H = (2L/N)(JᵀJ + S) at V, reusing the adjoint variable λ.  Closed-form because the only
# nonlinearity is cubic (F_uu = 6αu, diagonal).  At V_true the residual is 0 ⇒ S = 0 ⇒ H = Gauss-Newton.
function hessian_matrix(inst, V)
    n  = inst.p.N
    u  = Newton_solve(inst.p, V)
    D2 = real(ifft(-(inst.p.ks .^ 2) .* fft(Matrix{ComplexF64}(I, n, n), 1), 1))
    L  = -D2 .+ Diagonal(3 .* inst.α .* u .^ 2 .+ V)
    λ  = L \ (u .- inst.u_target); J = -(L \ Diagonal(u))
    S  = -6 .* inst.α .* (J' * Diagonal(u .* λ) * J) .- J' * Diagonal(λ) .- Diagonal(λ) * J
    Symmetric((2 * inst.p.L / n) .* (J' * J .+ S))
end

# κ = σ_max/σ_min via SVD, so it stays ≥ 0 even when H is singular (the ill case).
function spectrum(inst, V)
    σ = svdvals(Matrix(hessian_matrix(inst, V)))
    (; σ, κ = σ[1] / σ[end])
end

# ============================================================================
# Experiment 2 — Conditioning: well vs ill.
#   Why does recovery succeed on one and collapse on the other?  Hessian structure + spectrum + recovered V.
#   Export: log|H| heatmap (per instance), spectrum plot, recovered-V plot, κ table.  (ill: loss→0 but V unrecovered.)
# ============================================================================
function experiment_conditioning()
    df    = DataFrame(config = String[], κ = Float64[], pot_err = Float64[])
    spec  = plot(yscale = :log10, xlabel = "index", ylabel = "σ(H)", title = "Hessian spectrum", legend = :bottomleft)
    recov = plot(WELL.p.x, WELL.Vtrue; label = "V_true", ls = :dash, c = :black, xlabel = "x", ylabel = "V",
                 title = "recovered V (L-BFGS, unregularized)")
    for (name, inst) in (("well", WELL), ("ill", ILL))
        sp = spectrum(inst, inst.Vtrue)
        r  = runopt(inst, optimizer("lbfgs", 5000))
        H  = hessian_matrix(inst, inst.Vtrue)
        h  = heatmap(inst.p.x, inst.p.x, log10.(abs.(Matrix(H)) .+ 1e-300);
                     clims = (-16, -6), c = :viridis, yflip = true,
                     xlabel = "x'", ylabel = "x", title = "log|H| — $name")
        save_figure(DATA, "conditioning_hessian_$name", h)
        push!(df, (name, sp.κ, r.pot_err))
        plot!(spec, sort(sp.σ; rev = true); label = name)
        plot!(recov, inst.p.x, r.V; label = "V* $name")
    end
    save_figure(DATA, "conditioning_spectrum", spec)
    save_figure(DATA, "conditioning_recovered_V", recov)
    write_table(DATA, "conditioning", df)
    df
end

# ============================================================================
# Experiment 3 — Nonlinearity sweep (α).
#   How does α reshape primal cost and inverse conditioning?  L-BFGS only, well and ill.
#   Export: κ-vs-α plot + table (newton iters up, κ down — primal stiffens, inverse regularizes).
# ============================================================================
function experiment_alpha()
    αs  = [0.5, 1.0, 10.0, 50.0, 100.0, 200.0, 500.0]
    plt = plot(xscale = :log10, yscale = :log10, xlabel = "α", ylabel = "κ(H)", title = "conditioning vs α", legend = :topright)
    out = DataFrame[]
    for (name, ushape) in (("well", u_well), ("ill", u_ill))
        df = DataFrame(α = Float64[], newton = Int[], κ = Float64[], it_to_tol = Int[], pot_err = Float64[])
        for α in αs
            inst = instance(ushape, α)
            _, nit = Newton_solve(inst.p, inst.Vtrue; return_iters = true)
            r = runopt(inst, optimizer("lbfgs", 5000))
            push!(df, (α, nit, spectrum(inst, inst.Vtrue).κ, r.iters_to_tol, r.pot_err))
        end
        plot!(plt, αs, df.κ; marker = :circle, label = name)
        write_table(DATA, "alpha_$name", df); push!(out, df)
    end
    save_figure(DATA, "alpha_kappa", plt)
    out
end

# ============================================================================
# Experiment 4 — Regularization: R1 (TV) vs R2 (Tikhonov).
#   On the ill case with noisy data, which prior recovers V best, and at what β?
#   Export: pot_err-vs-β plot (optimal β = minimum), despiking plot, table.
# ============================================================================
function experiment_regularization(instance; noise = 1e-4, tag = "")
    inst = noisy(instance; noise = noise)
    βs   = 10.0 .^ range(-10, -2; length = 20)
    o    = optimizer("lbfgs", 1000)                  # bounded: noisy ⇒ no early-stop

    df    = DataFrame(method = String[], β = Float64[], misfit = Float64[], pot_err = Float64[])
    pe_pl = plot(xscale = :log10, yscale = :log10, xlabel = "β", ylabel = "pot_err", title = "recovery vs β", legend = :topleft)
    V_pl  = plot(inst.p.x, runopt(inst, o; record = false).V; label = "β=0 (spiky)", xlabel = "x", ylabel = "V", title = "despiking at optimal β")
    β_opt = Float64[]
    summary = DataFrame(method = String[], β = Float64[], misfit = Float64[], pot_err = Float64[])
    for (mname, mk) in (("R1(TV)", β -> R1(β = β)), ("R2(Tik)", β -> R2(β = β)))
        pe, mis, Vs = Float64[], Float64[], Vector{Float64}[]
        for β in βs
            r = runopt(inst, o; reg = mk(β), record = false)
            m, p = loss_of(inst, r.V), pot_err(inst, r.V)
            push!(df, (mname, β, m, p))
            push!(pe, p); push!(mis, m); push!(Vs, r.V)
            println("β = $β, method = $mname, pot_err = $(round(p, sigdigits = 3))")
        end
        i★ = argmin(pe); β_optim = βs[i★]
        push!(β_opt, β_optim)
        push!(summary, (mname, β_optim, mis[i★], pe[i★]))   # optimal row per method (compact report table)
        plot!(pe_pl, βs, pe; marker = :circle, label = mname)
        plot!(V_pl, inst.p.x, Vs[i★]; label = "$mname β*=$(round(β_optim, sigdigits = 2))")
    end
    plot!(V_pl, inst.p.x, inst.Vtrue; label = "V_true", ls = :dash, c = :black)
    save_figure(DATA, "regularization_poterr$tag", pe_pl)
    save_figure(DATA, "regularization_despiking$tag", V_pl)
    write_table(DATA, "regularization$tag", df)                  # full sweep (kept for the record)
    write_table(DATA, "regularization_summary$tag", summary)     # 2-row summary → tab:regularization
    df, β_opt...
end

# ============================================================================
# Experiment 4b — Direct algebraic inversion vs optimisation.
#   The forward PDE can be solved for V pointwise: F(u,V)=0 ⇒ V = (f + u'' - αu³)/u.
#   Exact for clean, strictly-positive u (well case), but the u'' term amplifies the
#   data noise (×k²), so on noisy data it is useless — that is what the optimiser fixes.
#   Export: V_true vs V_direct vs L-BFGS recovery, plus the clean-vs-noisy data (one figure).
# ============================================================================
direct_V(inst) = (inst.p.f .+ diff2_fourier(inst.u_target, inst.p.ks) .- inst.α .* inst.u_target .^ 3) ./ inst.u_target

function experiment_direct(inst_clean; reg = NoReg(), noise = 1e-6, file = "direct_vs_optim", budget = 1000)
    inst = noisy(inst_clean; noise = noise)
    Vd = direct_V(inst)
    V★ = runopt(inst, optimizer("lbfgs", budget); reg = reg, record = false).V
    
    lo, hi = extrema(vcat(inst.Vtrue, V★)); pad = max(2.0, hi - lo)
    pV = plot(inst.p.x, Vd; label = "V_direct = (f + u'' - αu³)/u", lw = 1, alpha = 0.7, c = 2,
              xlabel = "x", ylabel = "V", ylims = (lo - pad, hi + pad), title = "direct inversion vs L-BFGS recovery")
    plot!(pV, inst.p.x, V★;          label = "V* (L-BFGS)", lw = 2, c = 3)
    plot!(pV, inst.p.x, inst.Vtrue;  label = "V_true", ls = :dash, c = :black, lw = 2)

    # u panel: the perturbation that wrecks V_direct is invisible to the naked eye
    pu = plot(inst_clean.p.x, inst_clean.u_target; label = "u (clean)", c = :black, lw = 2,
              xlabel = "x", ylabel = "u", title = "data: clean vs noisy")
    plot!(pu, inst.p.x, inst.u_target; label = "u + noise", c = 2, lw = 1, alpha = 0.8)
    plt = plot(pV, pu; layout = (1, 2), size = (1000, 420))
    save_figure(DATA, file, plt)
    plt
end

# ============================================================================
# Experiment 5 — Animation: V₀ → V* with the matching state.
#   Watch L-BFGS transform V₀=1 into V*, and u(V) approach the target. Well (clean) and ill (despiked).
#   Export: gif per case. 
# ============================================================================
function experiment_animation(inst, file; optim = "lbfgs", reg = NoReg(), budget = 1000)
    _, xs = optimize(optimizer(optim, budget), V -> loss_of(inst, V; reg), V -> grad_of(inst, V; reg),
                 ones(N); history_step = 1, metric = copy)
    us = [Newton_solve(inst.p, V) for V in xs]
    function lims(vecs, ref)                          # constant y-axis across all frames ⇒ stable animation
        lo, hi = extrema(vcat(vecs..., ref))
        (lo - 0.05(hi - lo), hi + 0.05(hi - lo))
    end
    vlims, ulims = lims(xs, inst.Vtrue), lims(us, inst.u_target)
    # static figure of the converged state, for the report (saved independently of the gif encoder)
    final = plot(plot(inst.p.x, [xs[end] inst.Vtrue]; label = ["V*" "V_true"], ylabel = "V", ylims = vlims,
                      title = "final state — $(replace(file, ".gif" => ""))"),
                 plot(inst.p.x, [us[end] inst.u_target]; label = ["u(V*)" "u_target"], ylabel = "u", xlabel = "x", ylims = ulims),
                 layout = (2, 1))
    save_figure(DATA, replace(file, ".gif" => "_final"), final)
    anim = @animate for (k, (V, u)) in enumerate(zip(xs, us))
        plot(plot(inst.p.x, [V inst.Vtrue]; label = ["Vₖ" "V_true"], ylabel = "V", ylims = vlims, title = "iter $k / $(length(xs))"),
             plot(inst.p.x, [u inst.u_target]; label = ["u(Vₖ)" "u_target"], ylabel = "u", xlabel = "x", ylims = ulims), layout = (2, 1))
    end
    gif(anim, joinpath(DATA, file); fps = div(length(xs), 10), progress = false)
end

# ============================================================================
# Experiment 6 — Conditioning vs mesh resolution (κ vs N).
#   Is the ill-posedness intrinsic or a discretization artifact?  κ at V_true over a range of N.
#   Export: κ-vs-N plot + table.  (well: κ grows ~N⁴ ⇒ the continuum inverse problem is ill-posed;
#   ill: singular at every N — u's zeros are grid points regardless of resolution.)
# ============================================================================
function experiment_kappa_N()
    ns = [50, 100, 200, 400, 800]
    df = DataFrame(N = Int[], κ_well = Float64[], κ_ill = Float64[])
    for n in ns
        iw = instance(u_well, α₀; n = n)
        ii = instance(u_ill,  α₀; n = n)
        push!(df, (n, spectrum(iw, iw.Vtrue).κ, spectrum(ii, ii.Vtrue).κ))
    end
    plt = plot(xscale = :log10, yscale = :log10, xlabel = "N", ylabel = "κ(H)", title = "conditioning vs resolution", legend = :left)
    plot!(plt, ns, df.κ_well; marker = :circle, label = "well")
    plot!(plt, ns, df.κ_ill;  marker = :circle, label = "ill")
    save_figure(DATA, "kappa_N", plt)
    write_table(DATA, "kappa_N", df)
    df
end

# ============================================================================
# Run everything.
# ============================================================================
function main()
    setup_check()
    experiment_optimizers()
    experiment_conditioning()
    experiment_alpha()
    experiment_kappa_N()
    
    well_noisy = noisy(WELL, noise=1e-6)                       # well-conditioned but noisy => regularisation has a job to do
    other_ill = instance(u_ill, α₀; V_fn = V_spike)

    _, r1_opt, r2_opt = experiment_regularization(ILL)
    experiment_animation(WELL, "anim_well.gif")
    experiment_animation(ILL, "anim_ill.gif")
    experiment_animation(well_noisy, "anim_well_reg1.gif"; reg = R1(β = r1_opt))
    experiment_animation(well_noisy, "anim_well_reg2.gif"; reg = R2(β = r2_opt))
    experiment_animation(ILL, "anim_ill_reg1.gif"; reg = R1(β = r1_opt))
    experiment_animation(ILL, "anim_ill_reg2.gif"; reg = R2(β = r2_opt))
    experiment_animation(other_ill, "anim_spike_ill_reg1.gif"; reg = R1(β = r1_opt))
    experiment_animation(other_ill, "anim_spike_ill_reg2.gif"; reg = R2(β = r2_opt))
e

    experiment_direct(WELL; reg = R2(β = r2_opt), file = "direct_vs_optim_well")  # naive 1/u inversion vs L-BFGS (noises WELL internally)

    write_meta(DATA; packages = ["Optim"])
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end