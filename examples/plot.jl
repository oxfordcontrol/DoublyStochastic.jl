using PyCall, JLD2, SparseArrays, LinearAlgebra
plt = pyimport("matplotlib.pyplot")
transforms = pyimport("matplotlib.transforms")
gridspec = pyimport("matplotlib.gridspec")

function plot_region(D, region, gs)
    M = copy(D)
    M.nzval .= log.(M.nzval .+ 1)
    ax = plt.subplot(gs)
    ax.axis("off")
    base = ax.transData
    rot = transforms.Affine2D().rotate_deg(-90)
    ax.pcolormesh(Matrix(M[region, region]), cmap="hot", transform = rot + base)# , vmin=-1, vmax=1)
    ax.axis("equal")
end

@load "GM12878_combined_chr7_500kb_results.jld2"
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(3, 3)
gs.update(wspace=0.025, hspace=0.01)

fig.patch.set_visible(false)
resolution = 500000
region = 1:size(C, 1) # plot the whole region
@assert norm(C - C') <= 1e-9
plot_region(C, region, gs[1])
plot_region(X, region, gs[2])
plot_region(Y, region, gs[3])

@load "GM12878_combined_chr7_5kb_results.jld2"
resolution = 5000
region = Int(137.2*10^6/resolution+1):Int(137.8*10^6/resolution+1)
plot_region(C, region, gs[4])
plot_region(X, region, gs[5])
plot_region(Y, region, gs[6])

@load "GM12878_combined_chr7_1kb_results.jld2"
resolution = 1000
region = Int(137.55*10^6/resolution+1):Int(137.75*10^6/resolution+1)
plot_region(C, region, gs[7])
plot_region(X, region, gs[8])
plot_region(Y, region, gs[9])

fig.canvas.print_png("hit_full.png")