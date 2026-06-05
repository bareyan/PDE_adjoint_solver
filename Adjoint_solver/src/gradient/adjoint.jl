
function adjoint_gradient(p, V, u_target; β, R_grad)
    u = Newton_solve_fourier(p, V, 20)
    
    L = (result, du) -> result .= -diff2_fourier(du, p.ks) + 3 * p.α .* u .^2 .* du .+ V .* du
    op = LinearOperator(Float64, p.N, p.N, true, true, L)

    c = sum(3 * p.α .* u .^2 .+ V) / p.N
    P = (out, v) -> out .= real(ifft(fft(v) ./ (p.ks.^2 .+ c)))
    P_op = LinearOperator(Float64, p.N, p.N, true, true, P)

    λ, _ = cg(op, u - u_target, M = P_op, atol=1e-13, rtol=1e-11)
    grad = -2λ .* u
    return u, grad + β .* R_grad
end