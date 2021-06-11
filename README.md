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
N. Rontsis and P. Goulart.
Optimal Approximation of Doubly Stochastic Matrices.
International Conference on Artificial Intelligence and Statistics. PMLR, 2020.
```

## Documentation
Run:
```julia
solve(C, max_iterations; kwargs...) -> Matrix, SparseIterable/DenseIterable
```
imported from `src/admm_sparse.jl` for sparse problems or `src/admm_dense.jl` for dense problems.
