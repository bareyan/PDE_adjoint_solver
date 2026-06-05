β = 1e-10
function R_2(V) 
    return sum(diff_fourier(V, ks).^2)/N
end
function grad_R_2(V)
    return - 2 .* diff2_fourier(V, ks)/N
end

function R_1(V;eps=1e-8)
    sum(sqrt.(diff_fourier(V, ks).^2 .+ eps^2))/N
end

function grad_R_1(V; eps=1e-8)
    d1 = diff_fourier(V, ks)
    return -diff_fourier(d1 ./ (d1.^2 .+ eps^2), ks)/N
end

