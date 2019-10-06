using DataFrames, CSV
using Printf
include("../src/admm_sparse.jl")
include("../src/admm_dense.jl")
include("load_problems.jl")
include("../src/other_solvers.jl")

normalize_data = true
problem_set = :random # Other options :suitesparse or :random

parameters_set_suitespace = 1:100
parameters_set_ocean = [(h, d) for d in 10.0^-1.1:-0.1:-1.3 for h in pi./[8; 16; 32; 64]][:]
parameters_set_random = [(n, 0.001) for n in (100, 1000, 10000)]
parameters_set = Dict(zip((:suitesparse, :ocean, :random),
    (parameters_set_suitespace, parameters_set_ocean, parameters_set_random)))

df = DataFrame(name = String[], n = Int[], nnz = Int[],
    time = Float64[], time_gurobi = Float64[], 
    iter = Int[]
)

for parameter in parameters_set[problem_set]
    if problem_set == :suitesparse
        C, name = load_suitesparse(parameter)
    elseif problem_set == :ocean
        h = parameter[1]; d = parameter[2]
        C, name = load_ocean(h, d)
    elseif problem_set == :random
        name = "random"
        n = parameter[1]; sparsity = parameter[2]
        @show n
        C = sprandn(n, n, sparsity); C = (C + C')/2;
    end

    # Scaling
    @printf("Problem Statistics: min value: %.3f, max: %.3f, nnz: %d, rows: %d.\n",
        minimum(C.nzval), maximum(C.nzval), nnz(C), size(C, 1))
    if normalize_data
        @info "Prescaling matrix data to [0, 1] with nonzero diagonal."
        C = abs.(C)
        if maximum(C.nzval) > 1
            C = C./maximum(C)
        end
        C = C + 1e-16*I
    end

    t_gurobi = 0.0; t = 0.0; iter = 0
    t_gurobi = @elapsed X_gurobi = doubly_stochastic_gurobi(C)
    t = @elapsed X, data = solve(C, max_iterations=20000, sigma=5, rho=50, alpha=1.6, ε_abs=1e-4)
    @show norm(X - X_gurobi)
        
    push!(df, [name, size(C, 1), nnz(C), t, t_gurobi, iter])
    df |> CSV.write(string("../results/", problem_set, ".csv"))
end