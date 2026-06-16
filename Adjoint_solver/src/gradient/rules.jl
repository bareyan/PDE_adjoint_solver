# Custom differentiation rules:
#   * cg_ad       — a rule for the linear solve, so reverse-mode AD never unrolls CG.
#   * Newton_solve — a rule for the whole state solve, so reverse-mode AD composes
#                    the analytic adjoint instead of unrolling Newton.
# The rules are written once with ChainRulesCore; Zygote consumes them natively and
# Mooncake picks them up through @from_rrule.

using ChainRulesCore
using Mooncake
using LinearAlgebra
using Krylov, LinearOperators, FFTW

include("../utils.jl")

# --- linear solve: x = A \ b via cg_ad -------------------------------------
# Forward:  differentiate A x = b  ⇒  A ẋ = ḃ - Ȧ x.
# Reverse:  x̄ ↦ b̄ = Aᵀ \ x̄,  Ā = -b̄ xᵀ.

function ChainRulesCore.frule((_, ΔA, Δb), ::typeof(cg_ad),
                              A::AbstractMatrix, b::AbstractVector; M = I, kwargs...)
    x = cg_ad(A, b; M = M, kwargs...)
    b_dot = Δb isa AbstractZero ? zero(b) : Δb
    rhs = ΔA isa AbstractZero ? b_dot : b_dot - ΔA * x
    x_dot = cg_ad(A, rhs; M = M, kwargs...)
    return x, x_dot
end

function ChainRulesCore.rrule(::typeof(cg_ad),
                              A::AbstractMatrix, b::AbstractVector; M = I, kwargs...)
    x = cg_ad(A, b; M = M, kwargs...)
    function cg_ad_pullback(Δx)
        x_bar = unthunk(Δx)
        x_bar isa AbstractZero && return (NoTangent(), ZeroTangent(), ZeroTangent())
        b_bar = cg_ad(A', x_bar; M = M', kwargs...)
        A_bar = -b_bar * x'
        return (NoTangent(), A_bar, b_bar)
    end
    return x, cg_ad_pullback
end

Mooncake.@from_rrule(Mooncake.DefaultCtx,
                     Tuple{typeof(cg_ad), AbstractMatrix, AbstractVector}, true)

# --- state solve: u(V) with F(u, V) = 0 ------------------------------------
# The rule lives on the production Newton_solve, so a reverse-mode AD of any loss
# built on it composes the analytic adjoint instead of differentiating the solver.
# The naive AD paths use newton_ad (not Newton_solve), so they never pick this up.
# The pullback is the implicit-function adjoint:
#   F(u, V) = 0  ⇒  du/dV = -J⁻¹ ∂F/∂V,   ∂F/∂V = diag(u),   J symmetric.
#   V̄ = (du/dV)ᵀ ū = -u .* (J⁻¹ ū).
# This is exactly the linear solve the hand adjoint performs (see adjoint.jl).

function ChainRulesCore.rrule(::typeof(Newton_solve), p, V, kwargs...)
    u = Newton_solve(p, V, kwargs...)
    function solve_state_pullback(ū)
        u_bar = unthunk(ū)
        Jv = (out, du) -> out .= -diff2_fourier(du, p.ks) .+ 3 * p.α .* u .^ 2 .* du .+ V .* du
        J  = LinearOperator(Float64, p.N, p.N, true, true, Jv)
        c  = sum(3 * p.α .* u .^ 2 .+ V) / p.N
        Pv = (out, v) -> out .= real(ifft(fft(v) ./ (p.ks .^ 2 .+ c)))
        P  = LinearOperator(Float64, p.N, p.N, true, true, Pv)
        w, _ = cg(J, u_bar; M = P, atol = 1e-13, rtol = 1e-11)
        return (NoTangent(), NoTangent(), -u .* w)
    end
    return u, solve_state_pullback
end

Mooncake.@from_rrule(Mooncake.DefaultCtx,
                     Tuple{typeof(Newton_solve), Any, AbstractVector})
