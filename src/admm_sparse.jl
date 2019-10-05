using Printf
using LinearAlgebra, SparseArrays, SuiteSparse

function generate_cost(C)
	# Generates p such that diagm(0 => p) is the problem's hessian
	p = 2*ones(size(C.nzval))
	for j in 1:size(C, 2)
		for idx in C.colptr[j]:C.colptr[j+1]-1
			i = C.rowval[idx]
			if i == j
				p[idx] = 4
			end
		end
	end
	H = Diagonal(p)

	return H
end

function triangulize!(X)
    n = size(X, 1)
    @inbounds for j in 1:n, idx in X.colptr[j]:X.colptr[j+1]-1
        i = X.rowval[idx]
        if j == i
            X.nzval[idx] /= 2
        elseif j < i
            X.nzval[idx] =0
        end
    end
    return dropzeros!(X)
end

triangulize(X) = triangulize!(copy(X))

mutable struct SparseIterable{T, Ti}
    n::Int
    C::SparseMatrixCSC{T, Ti}
    F::SuiteSparse.CHOLMOD.Factor{T} # F::SparseMatrixCSC{T, Ti}
    H::Vector{T}
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
    print_interval::Ti
    primal_residuals::Vector{T}
    dual_residuals::Vector{T}
    update_rho::Bool

    function SparseIterable(C::SparseMatrixCSC{T, Ti}; rho=50.0, sigma=1.0, alpha=1.7, ε_abs=1e-4, ε_rel=0.0, print_interval=50) where {T, Ti}
        Cu = triangulize!(copy(C))
        n = size(Cu, 1)
        z = ones(n); z_ = ones(n);
        w = zeros(nnz(Cu)); y = zeros(n)
        x = copy(Cu.nzval)
        H = generate_cost(Cu).diag
        Cu.nzval .*= H

        new{T, Ti}(n, Cu, cholesky(sprandn(0, 0, 0.0)), H, 0*x, 0*copy(x), z, z_, w, y, sigma, rho, alpha, 0, ε_abs, ε_rel, 0.0,
            print_interval, zeros(T, 0), zeros(T, 0), true)
    end
end

function compute_factorization!(data::SparseIterable)
    S = copy(data.C); S.nzval .= 1
    S = (S + S')
    S.nzval ./= (2 + data.sigma)
    @inbounds for i in 1:size(S, 1)
        S[i, i] *= (2 + data.sigma)
        S[i, i] /= (4 + data.sigma)
    end
    S .*= data.rho
    sums = sum(S, dims=1)
    S = S + I
    @inbounds for i in 1:size(S, 1)
        S[i, i] += sums[i]
    end
    print("Computing cholesky...")
    data.F = cholesky(S)
    println("  Done!")
end

function print_info(data::SparseIterable)
    if mod(data.iteration, data.print_interval*10) == 0 || data.iteration == 0
        @printf("Iter \t  Objective \t Primal Res \t Dual Res \t Time (s) \n")
    end
    Ax = mul_A_sparse(data.C, data.x)
    r_prim = norm(Ax .- 1, Inf)
    r_dual = norm(data.H.*data.x - data.C.nzval + mul_At_sparse(data.C, data.y) + data.w, Inf)

    @printf("%d \t  %.5e \t %.5e \t %.5e \t %.5e\n",
        data.iteration,
        data.x'*(data.H.*data.x)/2 - dot(data.x, data.C.nzval),
        r_prim,
        r_dual,
        data.time
    )

    if r_prim <= data.ε_abs && r_dual <= data.ε_abs
        done = true
    else
        done = false
    end

    ratio = r_prim/r_dual
    if ratio > 20 && !done && data.update_rho
        # set_phase!(data.ps, Pardiso.RELEASE_ALL)
        # pardiso(data.ps)
        # data.ps = MKLPardisoSolver()
        data.sigma *= 10; data.rho *= 10
        compute_factorization!(data)
    end
    if ratio < 1/20 && !done && data.update_rho
        # set_phase!(data.ps, Pardiso.RELEASE_ALL)
        # pardiso(data.ps)
        # data.ps = MKLPardisoSolver()
        data.sigma /= 10; data.rho /= 10
        compute_factorization!(data)
    end
    
    return done
end

function admm_inner(data, max_iterations)
    done = print_info(data)
    while !done && data.iteration !== max_iterations
        data.time += @elapsed iterate!(data)
        if mod(data.iteration, data.print_interval) == 0
            done = print_info(data)
        end
        #=
        Ax = mul_A_sparse(data.C, data.x)
        r_prim = norm(Ax .- 1, Inf)
        r_dual = norm(data.H.*data.x - data.C.nzval + mul_At_sparse(data.C, data.y) + data.w, Inf)
        append!(data.primal_residuals, r_prim)
        append!(data.dual_residuals, r_dual)
        =#
    end
    return data
end

function solve(C::SparseMatrixCSC; max_iterations=5000, kwargs...)
    data = SparseIterable(C; kwargs...)
    compute_factorization!(data)
    admm_inner(data, max_iterations)
    # Extract solution
    X = copy(data.C)
    X.nzval .= data.x
    X = (X + X')
    return X, data
end

function solve(C::SparseMatrixCSC, data::SparseIterable; max_iterations=5000)
    data.iteration = 0
    data.C = triangulize!(copy(C))
    H = generate_cost(data.C).diag
    data.C.nzval .*= H

    compute_factorization!(data)
    data.update_rho = false
    admm_inner(data, max_iterations)
    return data
end

function iterate!(data::SparseIterable)
    solve_linear_system!(data)
    @. data.x_ = data.alpha*data.x_ + (1 - data.alpha)*data.x
    @. data.x = data.x_ + data.w/data.sigma
    project!(data)
    @. data.w += data.sigma*(data.x_ - data.x)
    @. data.y += data.rho*(data.alpha*data.z_ - data.alpha)
    data.iteration += 1
end

function project!(data::SparseIterable)
    @inbounds for i = 1:length(data.x)
        if data.x[i] <= 0
            data.x[i] = 0
        elseif data.x[i] >= 1
            data.x[i] = 1
        end
    end
end

function solve_linear_system!(data::SparseIterable)
    data.z .= data.rho .- data.y # Here data.z is used as a temporary variable
    data.z_ .= 0
    # Change to sparse representation
    @inbounds for j in 1:data.n
        for idx in data.C.colptr[j]:data.C.colptr[j+1]-1
            i = data.C.rowval[idx]
            data.x_[idx] = data.sigma*data.x[idx] - data.w[idx] + data.z[i] + data.z[j] + data.C.nzval[idx]
            # Equivalent to doing data.z_ .= A(data.x_) after the loop
            # data.z_[j] += data.x_[idx]
            # data.z_[i] += data.x_[idx]
        end
    end
    data.z .= mul_A_sparse(data.C, data.x_./(data.H .+ data.sigma))
    # set_phase!(data.ps, 33) # Solve, iterative refinement
    # pardiso(data.ps, data.z_, data.F, data.z)
    data.z_ .= data.F\data.z
    # Change to sparse representation
    @inbounds for j in 1:data.n
        for idx in data.C.colptr[j]:data.C.colptr[j+1]-1
            i = data.C.rowval[idx]
            data.x_[idx] = data.x_[idx] - data.rho*data.z_[i] - data.rho*data.z_[j] # ToDo: inverse
        end
    end
    data.x_ ./= (data.H .+ data.sigma)
end

function mul_A_sparse(C::SparseMatrixCSC, x)
    n = size(C, 1)

    y = zeros(n)
    @inbounds for j in 1:n, idx in C.colptr[j]:C.colptr[j+1]-1
        i = C.rowval[idx]
        y[i] += x[idx]
        y[j] += x[idx]
    end

    return y
end

function mul_At_sparse(C::SparseMatrixCSC, x)
    n = size(C, 1)
    y = zeros(nnz(C))
    @inbounds for j in 1:n, idx in C.colptr[j]:C.colptr[j+1]-1
        i = C.rowval[idx]
        y[idx] += x[i] + x[j]
    end

    return y
end