#############################################
### This file contains analysis of regularization methods for solving an inverse problem to find V(x) for the following equation
### $$ -u''(x) + αu(x)^3 + V(x)u(x) = f(x) $$
### with $u(0) = u(2π), u'(0) = u'(2π)$
### The basic loss function we are interested in is the following
### $$ L(V) = \int_0^{2π} (u(x) - u_{target}(x))^2 dx $$
### We will add regularization terms to the loss function to improve the stability of the inverse problem.
#############################################

#### Setup

#### Regularization methods

### Search for optimal regularization parameter: R1

### Search for optimal regularization parameter: R2

#### Comparison of regularization methods

