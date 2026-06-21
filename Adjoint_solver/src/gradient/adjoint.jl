function adjoint_gradient(p, V, u_target; reg::Regularizer=NoReg())
    u = Newton_solve(p, V)
    
    L = (result, du) -> result .= -diff2_fourier(du, p.ks) + 3 * p.α .* u .^2 .* du .+ V .* du
    op = LinearOperator(Float64, p.N, p.N, true, true, L)

    c = max(sum(3 * p.α .* u .^2 .+ V) / p.N, 1.0)   # floor > 0 keeps (ks²+c) preconditioner SPD
    P = (out, v) -> out .= real(ifft(fft(v) ./ (p.ks.^2 .+ c)))
    P_op = LinearOperator(Float64, p.N, p.N, true, true, P)

    λ, _ = cg(op, u - u_target, M = P_op, atol=1e-13, rtol=1e-11)
    grad = (p.L/p.N) .*  (-2λ .* u)
    return u, grad + grad_R(p, V, reg)
end