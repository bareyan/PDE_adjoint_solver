abstract type Regularizer end

Base.@kwdef struct R1 <: Regularizer
    β::Float64 = 0.0
    eps::Float64 = 1e-8
end

Base.@kwdef struct R2 <: Regularizer
    β::Float64 = 0.0
end

Base.@kwdef struct NoReg <: Regularizer
end

function R(p::Problem, V, reg::R1)
    return reg.β * (p.L/p.N) * sum(sqrt.(diff_fourier(V, p.ks).^2 .+ reg.eps^2))
end

function R(p::Problem, V, reg::R2)
    return reg.β * (p.L/p.N) * sum(diff_fourier(V, p.ks).^2)
end

function R(p::Problem, V, reg::NoReg)
    return 0.0
end

function grad_R(p::Problem, V, reg::R1)
    d1 = diff_fourier(V, p.ks)
    return -reg.β * (p.L/p.N) .* diff_fourier(d1 ./ sqrt.(d1.^2 .+ reg.eps^2), p.ks)
end

function grad_R(p::Problem, V, reg::R2)
    return - 2 * reg.β * (p.L/p.N) .* diff2_fourier(V, p.ks)
end

function grad_R(p::Problem, V, reg::NoReg)
    return zeros(size(V))
end
