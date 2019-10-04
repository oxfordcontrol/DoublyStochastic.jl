# DoublyStochastic.jl
This package solves the Doubly Stochastic Matrix Approximation Problem
```
minimize    ‖X - C‖
subject to  X is doubly stochastic
            X is zero at evey index where C is
```
where `C` is a given, symmetric matrix and `X` in the `n x n` matrix variable. The solution method is a custom [ADMM](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf) implementation that is efficient for both dense and sparse matrices C. In particular:
- When `C` is sparse, our solution method computes an initial Cholesky factorization of a matrix that has the same sparsity pattern as `C + I`, and then consists of operations of linear complexity with respect to the nonzeros in `C`.
- When `C` is dense, then the solution method merely consists of iterations of linear complexity.

The main reference for this package is
```
Rontsis, N., and Goulart, P.
Doubly Stochastic Matrix Approximation in the Frobenius Norm.
Preprint Submitted to arXiv
```

## Installation
This package can be installed by running
```
add https://github.com/oxfordcontrol/DoublyStochastic.jl
```
in [Julia's Pkg REPL mode](https://docs.julialang.org/en/v1/stdlib/Pkg/index.html#Getting-Started-1).
## Documentation
### Standard TRS
For small problems run:
```julia
doubly_stochastic(C, max_iterations; kwargs...) -> x, info
```
**Arguments** (`T` is any real numerical type):
* `P`: The quadratic cost represented as any linear operator implementing `mul!`, `issymmetric` and `size`.
* `q::AbstractVector{T}`: the linear cost.
* `r::T`: the radius.

**Output**
* `X::Matrix{T}`: Array with each column containing a global solution to the TRS
* `info::TRSInfo{T}`: Info structure. See [below](#the-trsinfo-struct) for details.

**Keywords (optional)**
* `tol`, `maxiter`, `ncv` and `v0` that are passed to `eigs` used to solve the underlying eigenproblem. Refer to `Arpack.jl`'s [documentation](https://julialinearalgebra.github.io/Arpack.jl/stable/) for these arguments. Of particular importance is **`tol::T`** which essentially controls the **accuracy** of the returned solutions.
* `tol_hard=2e-7`: Threshold for switching to the hard-case. Refer to [Adachi et al.](https://epubs.siam.org/doi/pdf/10.1137/16M1058200), Section 4.2 for an explanation.
* `compute_local::Bool=False`: Whether the local-no-global solution should be calculated. More details [below](#finding-local-no-global-minimizers).
