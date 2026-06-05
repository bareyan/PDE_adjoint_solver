## Residual
function F_fourier(u, p::FourierParams)
    return -diff2_fourier(u, p.Ks) + p.α .* u.^3 + p.V .* u - p.f
end

