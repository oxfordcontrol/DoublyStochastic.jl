include("../src/admm_sparse.jl")
include("../src/other_solvers.jl")
using CSV, DataFrames
using AMD, SparseArrays
using Statistics
using JLD2

function normalize(filename, resolution)
	# Load suggested scalings obtained via matrix balancing
	scaling_data = CSV.read(string(filename, ".KRnorm"), delim='\t', header=0) |> DataFrame
	weights = Vector{Float64}(scaling_data.Column1)
	n = length(weights)

	# Load pairwise contacts matrix
	data = CSV.read(string(filename, ".RAWobserved"), delim='\t', header=0) |> DataFrame
	ii = Vector{Int64}(data.Column1/resolution .+ 1)
	jj = Vector{Int64}(data.Column2/resolution .+ 1)
	v = Vector{Float64}(data.Column3)
	C = sparse(ii, jj, v)
	n = size(C, 1)
	if size(C, 1) < n 
		C = [C; spzeros(n - size(C, 1), size(C, 2))]
	end
	if size(C, 2) < n 
		C = [C spzeros(size(C, 1), n - size(C, 2))]
	end
	C = C + C'
	for i in 1:size(C, 1)
		C[i, i] /= 2
	end
	
	if true #all(isnan.(weights))
		C_ = C + scaling*1e-16*I;
		Y, d = bnewt(C_)
		@assert all(sum(Y, dims) .≈ 1)
		Y .*= scaling
		dropped_indices = zeros(n) .< 1 # all false
	else
		# Remove columns/rows that are too sparse to be scaled
		# In these columns, the suggested weights are NaN
		dropped_indices = isnan.(weights) # sum(C, dims=1)[:] .> 0
		C_ = C[.!dropped_indices, .!dropped_indices]
		scaling = sum(C_)/size(C_, 1)

		# # Construct scaled matrix obtained via matrix balancing
		Y = copy(C)
		@inbounds for j in 1:n, idx in Y.colptr[j]:Y.colptr[j+1]-1
			i = Y.rowval[idx]
			if !isnan(weights[i]) && !isnan(weights[j])
				Y.nzval[idx] /= (weights[i]*weights[j])
			end
		end
		@assert scaling ≈ mean(sum(Y[.!dropped_indices, .!dropped_indices], dims=1))	
	end
	
	# permutation = amd(C); C = C[permutation, permutation] # AMD appears to be of no use for this matrix :(
	t = @elapsed X_, data = solve(C_/scaling, ε_abs=1e-3) #max_iterations=20000, sigma=5, rho=50, alpha=1.6, ε_abs=1e-4)
	X_ .*= scaling
	X = spzeros(n, n)
	X[.!dropped_indices, .!dropped_indices] = X_
	@show norm(X - C)
	@show norm(Y - C)
	out_filename = string("GM12878_combined_", split(Base.Filesystem.basename(filename), '.')[1:end-1]..., "results.jld2")
	@show out_filename
	@save out_filename X Y C resolution dropped_indices
end

normalize("data/HIT/chr7_500kb.RAWobserved", 500000)
normalize("data/HIT/chr7_5kb.RAWobserved", 500000)
normalize("data/HIT/chr7_1kb.RAWobserved", 500000)