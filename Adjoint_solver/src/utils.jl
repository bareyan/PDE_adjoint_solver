using FFTW
## Utilities
### Returns a N equally distanced points from a to b.
function linspace(a::Number, b::Number, N::Integer)
    h = (b-a)/N
    collect(a:h:b-h)
end

function l2norm(v::Vector{Float64})
    return sqrt(sum(v.^2)/length(v))
end
function infnorm(v::Vector{Float64})
    return maximum(abs.(v))
end

## Derivatives
### First derivative using fft
function diff_fourier(u, ks)
    u_trans = fft(u)
    real(ifft(im .* ks .* u_trans))
end

### Second derivative using fft
function diff2_fourier(u, ks)
    real(ifft(- ks.^2 .* fft(u)))
end


## AD Friendly conjugate gradient
function cg_ad(applyA, b; M, n_iter, tol=1e-12)
    x  = zeros(eltype(b), length(b))
    r  = b - applyA(x)
    z  = M(r)
    p  = copy(z)
    rz = dot(r, z)
    for _ in 1:n_iter
        Ap  = applyA(p)
        pAp = dot(p, Ap)
        abs(pAp) < tol && break
        α   = rz / pAp
        x   = x + α .* p
        r   = r - α .* Ap
        z   = M(r)
        rz_new = dot(r, z)
        abs(rz) < tol && break
        β   = rz_new / rz
        p   = z + β .* p
        rz  = rz_new
    end
    return x
end