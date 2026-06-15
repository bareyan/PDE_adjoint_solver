include("utils.jl")
include("gradient/adjoint.jl")
include("gradient/simple.jl")

using Krylov, LinearOperators

## Newton solver
function Newton_solve(p::Problem, V;n_iter = 100, tol=1e-16)
    u = zeros(p.N) # u_0
    for i in 1:n_iter
        res = F(u; p=p, V=V)
        # Jacobian
        L = (result, du) -> result .= -diff2_fourier(du, p.ks) + 3 * p.α .* u .^2 .* du .+ V .* du
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
            # println("Stop condition at step $i")
            return u
        end
    end
    return u
end