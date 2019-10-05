using Printf

mutable struct DenseIterable{T}
    n::Int
    c::Vector{T}
    x::Vector{T}
    x_::Vector{T}
    z::Vector{T} # This not really used except as a temporary variable
    z_::Vector{T}
    w::Vector{T}
    y::Vector{T}
    sigma::T
    rho::T
    alpha::T
    iteration::Int
    ε_abs::T
    ε_rel::T
    time::T
    is_symmetric::Bool

    function DenseIterable(C::AbstractArray{T}; rho=10, sigma=50, alpha=1.7, ε_abs=1e-4, ε_rel=0.0) where T
        c = reshape(C, length(C))
        n = Int(sqrt(length(c)))
        z = ones(2*n); z_ = ones(2*n);
        w = zeros(n^2); y = zeros(2*n)
        C = reshape(c, n, n)
        X = (C + abs.(C))./2
        new{T}(n, c, X[:], X[:], z, z_, w, y, sigma, rho, alpha, 0, ε_abs, ε_rel, 0.0, issymmetric(C))
    end
end

function print_info(data::DenseIterable)
    if mod(data.iteration, 500) == 0 || data.iteration == 0
        @printf("Iter \t  Objective \t Primal Res \t Dual Res \t Time (s) \n")
    end
    X = reshape(data.x, data.n, data.n)
    Ax = mul_A_dense(data.x)
    r_prim = norm(Ax .- 1, Inf)
    r_dual = norm(data.x - data.c + mul_At_dense(data.y) + data.w, Inf)

    @printf("%d \t  %.5e \t %.5e \t %.5e \t %.5e\n",
        data.iteration,
        norm(data.x)^2/2 - dot(data.x, data.c),
        r_prim,
        r_dual,
        data.time
    )

    ε_prim = data.ε_abs + data.ε_rel*max(norm(Ax, Inf), 1)
    @inbounds y1 = view(data.y, 1:data.n)
    @inbounds y2 = view(data.y, data.n+1:2*data.n)
    ε_dual = data.ε_abs + data.ε_rel*max(norm(data.x, Inf), norm(y1, Inf)*norm(y2, Inf), norm(data.c, Inf), norm(data.w, Inf))
    if r_prim <= ε_prim && r_dual <= ε_dual
        done = true
    else
        done = false
    end

    return done
end

function solve(C::Matrix; max_iterations=5000, kwargs...)
    data = DenseIterable(C; kwargs...)
    done = print_info(data)
    while !done && data.iteration !== max_iterations
        data.time += @elapsed iterate!(data)
        if mod(data.iteration, 50) == 0
            done = print_info(data)
        end
    end
    return reshape(data.x, size(C, 1), size(C, 1)), data
end

function iterate!(data::DenseIterable)
    solve_linear_system!(data)
    @. data.x_ = data.alpha*data.x_ + (1 - data.alpha)*data.x
    @. data.x = data.x_ + data.w/data.sigma
    project!(data)
    @. data.w += data.sigma*(data.x_ - data.x)
    @. data.y += data.rho*(data.alpha*data.z_ - data.alpha)
    data.iteration += 1
end

function project!(data::DenseIterable)
    @inbounds for i = 1:length(data.x)
        if data.c[i] == 0 || data.x[i] <= 0
            data.x[i] = 0
        elseif data.x[i] >= 1
            data.x[i] = 1
        end
    end
end

function solve_linear_system!(data::DenseIterable)
    data.z .= data.rho .- data.y # Here data.z is used as a temporary variable
    data.z_ .= 0
    @inbounds for j = 1:data.n, i = 1:data.n
            idx = (j - 1)*data.n + i
            data.x_[idx] = data.sigma*data.x[idx] - data.w[idx] + data.z[i] + data.z[j + data.n] + data.c[idx]
            # Equivalent to doing data.z_ .= mul_A_dense(data.x_) after the loop
            # data.z_[j + data.n] += data.x_[idx]
            # data.z_[i] += data.x_[idx]
    end
    data.z_ .= mul_A_dense(data.x_, data.is_symmetric)
    solve_reduced_linear_system!(data.z_, data.rho, data.sigma)
    @inbounds for j = 1:data.n, i = 1:data.n
            idx = (j - 1)*data.n + i
            data.x_[idx] = (data.x_[idx] - data.rho*data.z_[i] - data.rho*data.z_[j + data.n])/(1 + data.sigma)
    end
end

function mul_A_dense(x, is_symmetric=false)
    n = Int(sqrt(length(x)))
    X = reshape(x, n, n)
    sum_rows = reshape(sum(X, dims=1), n)
    if is_symmetric
        sum_cols = sum_rows
    else
        sum_cols = reshape(sum(X, dims=2), n)
    end
    return [sum_cols; sum_rows]
end

function mul_At_dense(z)
    n = div(length(z), 2)
    z1 = view(z, 1:n)
    z2 = view(z, n+1:2*n)
    return reshape(z2' .+ z1, n^2)
end

function solve_reduced_linear_system!(x, rho, sigma)
    n = div(length(x), 2)
    c1 = 1/(1 + n*rho + sigma)
    c2 = c1^2*rho^2*n/(1 - c1^2*rho^2*n^2)
    c3 = -c1*rho*(1 + n*c2)

    @inbounds x1 = view(x, 1:n)
    @inbounds x2 = view(x, n+1:2*n)
    # s1 and s2 are always equal as they are the sum of the colum-sums and sum of the row-sums of a matrix
    s1 = sum(x1); s2 = sum(x2)

    lmul!(c1, x)
    c = c1*(c2*s1 + c3*s2)
    x1 .+= c
    c = c1*(c3*s1 + c2*s2)
    x2 .+= c
    # Both of the c above are equal, due to the same argument as for s1 and s2

    return x
end