using Enzyme
using FFTW
using Krylov, LinearOperators

## Utilities

### Returns a N equally distanced points from a to b.
function linspace(a::Number, b::Number, N::Integer)
    h = (b-a)/N
    collect(a:h:b-h)
end

## Derivatives

### First derivative calculation matrix
function diff(N)
    # We need to calculate the u', which is just a linear operator in this case, so we construct a matrix to do that
    D = zeros(N, N)
    h = 1/N
    for i in 1:N
        D[i, mod(i, N)+1] = 1/2h
        D[i, mod(i-2, N)+1 ] = -1/2h
    end
    return D
end

### Second derivative calculation matrix
function diff2(N)
    D2 = zeros(N, N)
    h = 1/N
    for i in 1:N
        D2[i, i] = -2/(h^2)
        D2[i, mod(i, N)+1] = 1/(h^2)
        D2[i, mod(i-2, N) +1] = 1/(h^2)
    end
    return D2
end

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

struct SimpleParams
    N::Int64
    α::Float64
    V::Vector{Float64}
    f::Vector{Float64}
    D2::Matrix{Float64}
end

struct FourierParams
    N::Int64
    α::Float64
    V::Vector{Float64}
    f::Vector{Float64}
    Ks::Vector{Float64}
end

### Create simpleParam object
function create_params(N)
    xx = linspace(0, 1, N)

    u_exact = sin.(2pi .* xx)
    # α - a real
    α = 1.
    # V(x) - a function
    V = 1.0 .+ 0.5 .* cos.(2π .* xx)
    #f(x) - a function
    f = 4π^2*sin.(2π .* xx) + α .* u_exact.^3 + V .* u_exact
    
    return SimpleParams(N, α, V, f, diff2(N))
end

### Create Param object for fourier solve
function create_fparams(N, α,  v, f_fn)
    xx = linspace(0, 1, N)
    V = v(xx)
    f = f_fn(xx)
    
    return FourierParams(N, α, V, f,  collect(2π .* fftfreq(N, N)))
end

## Residual
function F(u, p::SimpleParams)
    return -p.D2* u .+ p.α .* u.^3 + p.V .* u - p.f
end

function F_fourier(u, p::FourierParams)
    return -diff2_fourier(u, p.Ks) + p.α .* u.^3 + p.V .* u - p.f
end


## Solvers
function Newton_solve(p::SimpleParams, method, n_iter)
    u = zeros(p.N) # u_0
    for i in 1:n_iter
        res = F(u, p)
        J = method(F, p, u)
        step = J \ res
        u = u - step
    end
    return u
end

function finite_differences(F, p, u; eps=1e-8) 
    J = zeros(p.N, p.N)
    for i in 1:p.N
        v = zeros(p.N)
        v[i] = 1
        J[:, i] = (F(u .+ eps * v, p) - F(u, p)) / eps
    end
    return J
end

function forward_AD(F, p, u)
    J = zeros(p.N, p.N)
    for i in 1:p.N
        v = zeros(p.N)
        v[i] = 1
        (col,) = autodiff(Forward, F, Duplicated(u, v), Const(p))
        J[:, i] = col
    end
    # J = jacobian(Forward, F, Duplicated(u), Const(p), chunk=Val(8))
    # print(J)
    return J
end

function F_i(u, p, i, F)
    F(u, p)[i]
end

function backward_AD(F, p, u)
    J = zeros(p.N, p.N)
    for i in 1:p.N
        v = zeros(p.N)
        autodiff(Reverse, F_i, Active, Duplicated(u, v), Const(p), Const(i), Const(F))
        J[i, :] = v
    end
    # J = jacobian(Forward, F, Duplicated(u), Const(p), chunk=Val(8))
    # print(J)
    return J
end

# function Newton_solve_fourier(p::FourierParams, n_iter; verbose=false, tol=1e-16)
#     u = zeros(p.N) # u_0
#     for i in 1:n_iter
#         res = F_fourier(u, p)
#         L = (result, du) -> result .= -diff2_fourier(du, p.Ks) + 3 * p.α .* u .^2 .* du .+ p.V .* du
#         op = LinearOperator(Float64, p.N, p.N, true, true, L)
#         step, stats = cg(op, res)
#         if(verbose)
#             println("Iteration: ", i,",\n stats: ", stats )
#         end
#         u = u - step
#         if(sum(step.^2)<tol)
#             if(verbose)
#                 println("Took ", i, "iterations to converge")
#             end
#             return u
#         end
#     end
#     return u
# end
