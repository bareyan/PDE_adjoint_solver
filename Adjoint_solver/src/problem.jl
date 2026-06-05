include("utils.jl")

struct Problem
    L::Float64
    N::Int64
    x::Vector{Float64}
    ks::Vector{Float64}
    α::Float64
    f::Vector{Float64}
end

function Problem(;N::Int, L::Float64=2π, α::Float64=1., f)
    x  = collect(range(0, L; length = N + 1))[1:N]
    ks = wavenumbers(bc, N, L)
    fv = f isa Function ? f.(x) : f
    Problem(L, N, x, ks, α, fv)
end

function F(u; p::Problem, V)
    return -diff2_fourier(u, p.ks) + p.α .* u.^3 + V .* u - p.f
end

