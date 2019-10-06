include("../src/admm_sparse.jl")
using CSV, DataFrames
using AMD, SparseArrays
using Statistics
using JLD2

# Load pairwise contacts matrix
data = CSV.read("data/HIT/chr14_5kb.RAWobserved", delim='\t', header=0) |> DataFrame
ii = Vector{Int}(data.Column1/5000 .+ 1)
jj = Vector{Int}(data.Column2/5000 .+ 1)
v = Vector{Float64}(data.Column3)
C = sparse(ii, jj, v)
C = C + C'
for i in 1:size(C, 1)
	C[i, i] /= 2
end
# Load suggested scalings obtained via matrix balancing
scaling_data = CSV.read("data/HIT/chr14_5kb.KRnorm", delim='\t', header=0) |> DataFrame
weights = Vector{Float64}(scaling_data.Column1)

# Remove columns/rows that are too sparse to be scaled
# In these columns, the suggested weights are NaN
weights = weights[1:size(C, 1)]
dropped_indices = isnan.(weights) # sum(C, dims=1)[:] .> 0
C = C[.!dropped_indices, .!dropped_indices]
weights = weights[.!dropped_indices]
n = size(C, 1)
# Construct scaled matrix obtained via matrix balancing
Y = copy(C)
@inbounds for j in 1:n, idx in Y.colptr[j]:Y.colptr[j+1]-1
	i = Y.rowval[idx]
	if !isnan(weights[i]) && !isnan(weights[j])
		Y.nzval[idx] /= (weights[i]*weights[j])
	end
end
@save "matrix.jld2" Y
@assert false
scaling = sum(C)/n
@assert scaling ≈ mean(sum(Y, dims=1))
# permutation = amd(C); C = C[permutation, permutation]
t = @elapsed X, data = solve(C/scaling) #max_iterations=20000, sigma=5, rho=50, alpha=1.6, ε_abs=1e-4)
X .*= scaling
# include("solvers.jl")
# X, d = bnewt(C)
@show norm(X - C)
@show norm(Y - C)