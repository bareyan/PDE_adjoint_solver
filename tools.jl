using Enzyme
using FFTW
using Krylov, LinearOperators

## Utilities
### Returns a N equally distanced points from a to b.
function linspace(a::Number, b::Number, N::Integer)
    h = (b-a)/N
    collect(a:h:b-h)
end

function l2norm(v::Vector{Float64})
    return sqrt(sum(v.^2)/length(v))
end
function infnorm(v::Vector{Float64})
    return maximum(abs.(v))
end

## Derivatives
### First derivative using fft
function diff_fourier(u, ks)
    u_trans = fft(u)
    real(ifft(im .* ks .* u_trans))
end

### Second derivative using fft
function diff2_fourier(u, ks)
    real(ifft(- ks.^2 .* fft(u)))
end

## Settings
struct FourierParams
    N::Int64
    α::Float64
    V::Vector{Float64}
    f::Vector{Float64}
    Ks::Vector{Float64}
end
## Residual
function F_fourier(u, p::FourierParams)
    return -diff2_fourier(u, p.Ks) + p.α .* u.^3 + p.V .* u - p.f
end


## Solvers
function Newton_solve_fourier(p::FourierParams, n_iter; tol=1e-16)
    u = zeros(p.N) # u_0
    for i in 1:n_iter
        res = F_fourier(u, p)
        L = (result, du) -> result .= -diff2_fourier(du, p.Ks) + 3 * p.α .* u .^2 .* du .+ p.V .* du
        op = LinearOperator(Float64, p.N, p.N, true, true, L)
        c = sum(3 * p.α .* u .^2 .+ p.V) / p.N
        P = (out, v) -> out .= real(ifft(fft(v) ./ (p.Ks.^2 .+ c)))
        P_op = LinearOperator(Float64, p.N, p.N, true, true, P)

        # step, stats = cg(op, res)
        step, _ = cg(op, res, M=P_op)
        u = u - step
        if(l2norm(step)<tol)
            return u
        end
    end
    return u
end