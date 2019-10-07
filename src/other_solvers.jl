using LinearAlgebra, SparseArrays
using Gurobi
using MATLAB

function generate_equality_constraints(C; symmetry=false)
    # Form A := [A1; A2]
    nzval1 = ones(nnz(C))
    colptr1 = Vector(1:nnz(C)+1)
    rowval1 = copy(C.rowval)
    A1 = SparseMatrixCSC(size(C, 1), nnz(C), colptr1, rowval1, nzval1)

    nzval2 = ones(nnz(C))
    colptr2 = Vector(1:nnz(C)+1)
    rowval2 = copy(C.rowval)
    Ct = SparseMatrixCSC(C')
    d = diff(C.colptr)
    idx = 1
    @inbounds for i = 1:length(d)
        rowval2[idx:idx + d[i] - 1] .= i
        idx += d[i]
    end
    A2 = SparseMatrixCSC(size(C, 2), nnz(C), colptr2, rowval2, nzval2)

    if symmetry
        return A1 + A2
    else
        return [A1; A2]
    end
end

function doubly_stochastic_gurobi(C; scaling=1.0)
    @time A = generate_equality_constraints(C)
    m, n = size(A)

    env = Gurobi.Env()
    setparam!(env, "Threads", 1)
    model = gurobi_model(env, H=SparseMatrixCSC(1.0*I, n, n), f=-scaling*C.nzval, Aeq=A, beq=scaling*ones(m), lb=zeros(n))
    optimize(model)
    sol = get_solution(model)
    X = copy(C)
    X.nzval .= sol
    return X
end

function alternating_projections(C::Matrix; maxiters=5000, tol=1e-4)
    X = copy(C)
    n = size(X, 1)
    iter = 0
    residuals = zeros(maxiters)
    row_sum = sum(X, dims=1); total_sum = sum(row_sum)
    while norm(row_sum .- 1, Inf) > tol && iter < maxiters
        print(iter, " ")
        X .+= 1/n*(1 .+ sum(row_sum)/n .- row_sum' .- row_sum);
        @. X = max(0, X);
        iter += 1

        row_sum = sum(X, dims=1); total_sum = sum(row_sum)
        residuals[iter] = norm(row_sum .- 1, Inf)
    end
    return X, residuals[1:iter]
end


function bnewt(C)
    X = copy(C)
    d = mxcall(:bnewt, 1, abs.(X))
    D = copy(X.nzval)
    n = size(X, 1)
    @inbounds for j in 1:n, idx in X.colptr[j]:X.colptr[j+1]-1
        i = X.rowval[idx]
        X.nzval[idx] *= d[i]*d[j]
        D[idx] = d[i]*d[j]
    end
    return X, d
end