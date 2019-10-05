using MAT, Glob
using CSV, DataFrames, Distances
using JLD2
using Printf

function load_suitesparse(i)
    working_dir = pwd()
    path = "./WeightedGraphs/"
    cd(path)
    files = glob("*.mat")
    cd(working_dir)
    if i > length(files)
        @assert false "No more files..."
    end
        file = files[i]
    matlab_file = matopen(string(path, file))
    C = read(matlab_file, "Problem")["A"]
    close(matlab_file)
    println("Loading ", file, " of size: ", size(C), " and nonzeros: ", nnz(C))
    return C, file
end

function load_spambase(sigma=20.0)
    D = convert(Matrix, CSV.File("data/spambase/spambase.data") |> DataFrame)
    C = pairwise(Euclidean(), D, dims=1);
    C = exp.(-C.^2 ./ sigma^2)
    C[abs.(C) .< 1e-7] .= 0
    C = sparse(C) # + 1e-16*ones(size(C)))

    return C, "spambase"
end

function load_ocean(h=pi/8, d=1e-3)
    #=
    N = 128*256
    x = rand(N, 2)
    x[:, 1] = 2*pi*x[:, 1]
    x[:, 2] = 4*pi*x[:, 2] .- pi
    D = pairwise(Euclidean(), x, dims=1)
    D .+= 1e-20
    D[D .> pi/8] .= 0
    D = SparseMatrixCSC(D)
    @save "ocean.jld2" D
    =#
    @load "data/Ocean/ocean.jld2"
    @assert h <= pi/8

    D.nzval[D.nzval .> h] .= 0
    dropzeros!(D)
    D.nzval .= exp.(-D.nzval./(4*d*0.1))
    @show h
    @printf("%.2E", h)
    return D, string("ocean_h", @sprintf("%.2E", h), "_d", @sprintf("%.2E", d))
end