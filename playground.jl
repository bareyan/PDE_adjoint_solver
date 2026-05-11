## Problem
# Solve -u''(x) + αu(x)^3 + V(x)u(x) = f(x)
# with u(0) = u(1), u'(0) = u'(1)  
# We formulate an equivalent problem: 
# Minimize a functional for which -u''(x) + αu(x)^3 + V(x)u(x) - f(x) is the gradient. 

# Why: The optimality requires the gradient to be 0, so when we optimize the functional, we get a 0 gradient 'point', which means -u''(x) + αu(x)^3 + V(x)u(x) - F(x).

## What is the functional to optimize: 
# E(u) = /int_0^1 [1/2 * u'(x)^2 + α/4 * u(x)^4 + V(x)/2 * u(x)^2 - f(x)u] dx

## Hyperparameters

# N - the space discretization bin count
N = 100
# Returns a N equally distanced points from a to b.
function linspace(a::Number, b::Number, N::Integer)
    h = (b-a)/N
    collect(a:h:b-h)
end

xx = linspace(0, 1, N)
## Problem parameters

# u - unknown to find
u_exact = sin.(2pi .*xx)

# α - a real
α = 1

# V(x) - a function
V = 1.0 .+ 0.5 .* cos.(2π .* xx)

# f(x) - a function
# We use the u_exact and the other parameters to find an f that verifies the pde
# f(x) = -u''(x) + αu'(x)^3 + V(x)u(x) = 4π^2sin(2πx) + 8π^3 cos(2πx) + sin(2πx) + 0.5cos(2πx)sin(2πx)
f = 4π^2*sin.(2π .* xx) + α .* u_exact.^3 + V .* u_exact

## Visualizations
using Plots

u_plot = plot(xx, u_exact, label="exact solution u(x)")
V_plot = plot(xx, [V zeros(N)], 
    labels = ["V(x)" "0"]
)
f_plot = plot(xx, f, label = "f(x)")




## Energy functional
# E(u) = /int_0^1 [1/2 * u'(x)^2 + α/4 * u(x)^4 + V(x)/2 * u(x)^2 - f(x)u] dx
e = 1/2 * (2π .* cos.(2π .*xx)).^2 + α/4 .* u_exact.^4 + V./2 .* u_exact.^2 - f .* u_exact
E = sum(e)/N 
energy_plot = plot(xx, e, label="energy",
fillrange = 0, fillalpha = 0.3, fillcolor = :orange, color= :orange)


plot(u_plot, V_plot, f_plot, energy_plot, layout=(2,2), titles=["u(x)" "V(x)" "f(x)" "Energy fn(optimal)"])
## Optimization 

# From here we suppose u in unknown and try to find a u that optimizes the functional
# Because of the discretization it comes back to a N dimensional optimization problem

# Astuces:
# u' can be calculated the following ways: 
    # u[1:] - u where the indexing wraps around, as we have the u(0)= u(1) condition
    # u'(0) = u'(1) condition prompts us to have derivative equal from both ways approaching so we can have 
        # u' = (u[1:] - u[-1:] ) / 2h

    # Spectral approach
    # We use the DFT, and the derivative in fourier space is easier to calculate
## Derivatives 
function diff(u)
    # We need to calculate the u', which is just a linear operator in this case, so we construct a matrix to do that
    D = zeros(N, N)
    h = 1/N
    for i in 1:N
        D[i, mod(i, N)+1] = 1/2h
        D[i, mod(i-2, N)+1 ] = -1/2h
    end
    return D
end
du_calc = diff(u_exact) * u_exact
du_exact = 2π .* cos.(2π .* xx)
# du_calc_plot =plot(xx, du_calc)
# du_exact_plot = plot(xx, du_exact) 
du_plot = plot(xx, [du_calc du_exact], labels=["Calculated derivative" "Analytic derivative"], color=[:yellow :green], ls= [:solid :dot])
# plot(du_calc_plot, du_exact_plot, du_plot, layout = (2, 2))
## Second derivative
function diff2(u)
    D2 = zeros(N, N)
    h = 1/N
    for i in 1:N
        D2[i, i] = -2/h^2
        D2[i, mod(i, N)+1] = 1/h^2
        D2[i, mod(i-2, N) +1] = 1/h^2
    end
    return D2
end
d2u = diff(du_calc) * du_calc
d2u_form = diff2(u_exact) * u_exact
d2u_exact = -4π^2 .* sin.(2π .* xx)
plot(xx, [d2u d2u_form d2u_exact], labels=["Calculated 2nd derivative" "Matrix method" "Analytic 2nd derivative"], color=[:yellow :blue :green], ls= [:solid :dash :dot])
