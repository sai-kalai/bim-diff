# Plot scalar variable (error or solution) on a dense grid inside the
# domain


using StaticArrays

using BoundaryIntegralEquations
using BoundaryIntegralEquations.DevTools



n_dense = 60
Γ_dense = DiscreteClosedCurve(n_dense, starfish)
xmin, xmax, ymin, ymax = extrema(Γ_dense)
xs = range(xmin, xmax, length=n_dense)
ys = range(ymin, ymax, length=n_dense)
iter = Iterators.product(xs, ys)
x_dense = stack(((x, y),) -> SA[x, y], iter; dims=2)


Γ_source, density_source, u_exact = manufactured_solution(Laplace(), x_dense)

Γ = DiscreteClosedCurve(n, starfish)

res = run_all_simulations(
    x_dense,
    ;
    n_vals=[100, 200, 400,],
    cutoff_vals=[0., 0.05, 0.1,],
    approach_types=[Direct,],
    bc_types=[Dirichlet,],
    fd_acc_vals=[32,],
)

# TODO: separate acquire vs plot




