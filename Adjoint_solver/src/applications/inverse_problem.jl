function loss(V; p::Problem, u_target, reg::Regularizer=NoReg())
    u = Newton_solve(p, V)
    return (p.L/p.N) * sum((u - u_target).^2) + R(p, V, reg)
end
 
function inverse_problem(p::Problem, u_target; V0=ones(p.N),
                         optimizer::Optimizer=OLBFGS(), reg::Regularizer=NoReg())
    loss_fn = V -> loss(V; p=p, u_target=u_target, reg=reg)
    grad_fn = V -> adjoint_gradient(p, V, u_target; reg=reg)[2]
    return optimize(optimizer, loss_fn, grad_fn, V0)
end