## Returns a N equally distanced points from a to b.
function linspace(a::Number, b::Number, N::Integer)
    h = (b-a)/N
    collect(a:h:b-h)
end

## First derivative calculation matrix
function diff(N)
    # We need to calculate the u', which is just a linear operator in this case, so we construct a matrix to do that
    D = zeros(N, N)
    h = 1/N
    for i in 1:N
        D[i, mod(i, N)+1] = 1/2h
        D[i, mod(i-2, N)+1 ] = -1/2h
    end
    return D
end

## Second derivative calculation matrix
function diff2(N)
    D2 = zeros(N, N)
    h = 1/N
    for i in 1:N
        D2[i, i] = -2/(h^2)
        D2[i, mod(i, N)+1] = 1/(h^2)
        D2[i, mod(i-2, N) +1] = 1/(h^2)
    end
    return D2
end