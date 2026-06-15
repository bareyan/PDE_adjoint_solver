using ChainRulesCore
using Enzyme
using LinearAlgebra

include("../utils.jl")

function ChainRulesCore.frule((_, ΔA, Δb), ::typeof(cg_ad),
                              A::AbstractMatrix, b::AbstractVector; M = I, kwargs...)
    x = cg_ad(A, b; M = M, kwargs...)
    b_dot = Δb isa AbstractZero ? zero(b) : Δb
    rhs = if ΔA isa AbstractZero
        b_dot
    else
        b_dot - ΔA * x
    end
    x_dot = cg_ad(A, rhs; M = M, kwargs...)
    return x, x_dot
end

function ChainRulesCore.rrule(::typeof(cg_ad),
                              A::AbstractMatrix, b::AbstractVector; M = I, kwargs...)
    x = cg_ad(A, b; M = M, kwargs...)

    function cg_ad_pullback(Δx)
        x_bar = unthunk(Δx)
        if x_bar isa AbstractZero
            return (NoTangent(), ZeroTangent(), ZeroTangent())
        end

        b_bar = cg_ad(A', x_bar; M = M', kwargs...)
        A_bar = -b_bar * x'
        return (NoTangent(), A_bar, b_bar)
    end

    return x, cg_ad_pullback
end
