#################################
### This file contains comparaison of different first order methods of optimization for solving an inverse problem to find V(x) for the following equation
### $$ -u''(x) + αu(x)^3 + V(x)u(x) = f(x) $$
### with $u(0) = u(2π), u'(0) = u'(2π)$
### The gradient we are interested in is the gradient of the following loss function with respect to V(x)
### $$ L(V) = \int_0^{2π} (u(x) - u_{target}(x))^2 dx $$
### Refer to the `gradient)computation.jl` file for the comparison of different methods of gradient computation. We will use the rule based adjoint method for the optimization in this file.
##################################

#### Setup

#### Optimizers

### Gradient Descent

### Gradient Descent with Adam

### L-BFGS

#### Comparison

