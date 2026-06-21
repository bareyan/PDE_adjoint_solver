## Newton solver
function Newton_solve(p::Problem, V; n_iter = 1000, tol = 1e-11, return_iters = false)
    u = zeros(p.N) # u_0
    for i in 1:n_iter
        res = F(u; p=p, V=V)
 
        # convergence check on the residual, before taking another step
        if l2norm(res) < tol
            return return_iters ? (u, i - 1) : u
        end
 
        # Jacobian
        L = (result, du) -> result .= -diff2_fourier(du, p.ks) + 3 * p.α .* u .^2 .* du .+ V .* du
        op = LinearOperator(Float64, p.N, p.N, true, true, L)
        # Preconditioning
        c = max(sum(3 * p.α .* u .^2 .+ V) / p.N, 1.0)   # floor > 0 keeps (ks²+c) preconditioner SPD
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
    end
    return return_iters ? (u, n_iter) : u
end