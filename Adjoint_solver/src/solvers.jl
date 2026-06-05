include("utils.jl")
include("gradient/adjoint.jl")
include("gradient/simple.jl")

using Krylov, LinearOperators

## Newton solver
function Newton_solve(p::Problem, V,  n_iter; tol=1e-16)
    u = zeros(p.N) # u_0
    for i in 1:n_iter
        res = F(u; p=p, V=V)
        # Jacobian
        L = (result, du) -> result .= -diff2_fourier(du, p.ks) + 3 * p.α .* u .^2 .* du .+ p.V .* du
        op = LinearOperator(Float64, p.N, p.N, true, true, L)
        # Preconditioning
        c = sum(3 * p.α .* u .^2 .+ V) / p.N
        P = (out, v) -> out .= real(ifft(fft(v) ./ (p.ks.^2 .+ c)))
        P_op = LinearOperator(Float64, p.N, p.N, true, true, P)

        # Step calculation
        step, _ = try 
            cg(op, res, M=P_op, atol=1e-13, rtol=1e-11) 
        catch _ 
            minres(op, res, M=P_op) 
        end
        # Step
        u = u - step

        #Stop condition
        if(l2norm(step)<tol)
            return u
        end
    end
    return u
end


## AD Friendly Newton solver
function Newton_solve_ad(p::Problem, V, n_iter; tol=1e-16)
    u = zeros(N)
    for _ in 1:n_iter
        res = -D2 * u .+ p.α .* u.^3 .+ V .* u .- p.f
        applyJ    = δu -> -D2 * δu .+ (3α .* u.^2 .+ V) .* δu
        applyMinv = z  -> Minv * z
        step = cg_ad(applyJ, applyMinv, res, 50)
        u = u - step
    end
    return u
end