# Plot solution error on a dense grid inside the domain for different values
# of cutoff and discretization, for fixed quadrature order
# domain

using GLMakie
using StaticArrays
using JLD2

using BoundaryIntegralEquations
using BoundaryIntegralEquations.DevTools



const FILE = "dense"
const DATAFILE = joinpath("data", FILE * ".jld2")


# acquire data
n = 200
Γ_dense = DiscreteClosedCurve(n, starfish)
xmin, xmax, ymin, ymax = extrema(Γ_dense)
xs = range(xmin, xmax, length=n)
ys = range(ymin, ymax, length=n)
iter = Iterators.product(xs, ys)
x_dense = stack(((x, y),) -> SA[x, y], iter; dims=2)

res = if false && isfile(DATAFILE)
    @info "loaded `res` from $(DATAFILE)"
    load_object(DATAFILE)
else
    r = run_all_simulations(
        x_dense,
        ;
        n_vals=[100, 200, 400,],
        cutoff_vals=[0., 0.05, 0.1, 0.5,],
        approach_types=[Indirect,],
        bc_types=[Dirichlet,],
        fd_acc_vals=[32,],
    )
    save_object(DATAFILE, r)
    @info "saved `res` to $(DATAFILE)"
    r
end

valid_sols = [
    (k, sol) for (k, group) in res.solutions if (k.solution_t <: BVPSolution && k.correction isa Zeta)
    for sol in solutions(group)
]

sort!(valid_sols; by=begin

    ((k, sol),) -> cutoff(k.evalmethod)

end
)

n_sols = length(valid_sols)
n_cols = ceil(Int, sqrt(n_sols))
n_rows = ceil(Int, n_sols / n_cols)

fig = Figure(size=(300 * n_cols, 300 * n_rows))


@show length(valid_sols)

for (i, (k, sol)) in enumerate(valid_sols)

    row = div(i - 1, n_cols) + 1
    col = rem(i - 1, n_cols) + 1

    ax = Axis(
        fig[row, col][1, 1];
        # aspect=DataAspect()
    )
    curve = bvp(sol).boundary

    val = log10.(abs.(sol.u - res.u_exact) .+ eps(eltype(sol.u)))
    # reshape to plot in contourf

    @show cutoff(k.evalmethod)

    msk = mask(curve, x_dense, cutoff(k.evalmethod))

    val = reshape(val, (n, n))

if ! (cutoff(k.evalmethod) == 0.)
    val[.!msk].=NaN
    end

    @show sum(msk)

    # fig, ax = visualize(Γ_dense, false, false)

    # lo, hi = extrema(val[.! outside_mask])
    # step = (hi-lo) < 5 ? 0.5 : 1
    # levels = range(floor(lo), ceil(hi), step=step)


    co = contourf!(
        ax,
        curve,
        xs,
        ys,
        val,
        levels=10,
        extendlow=:auto,
        extendhigh=:auto,
    )

    sc0 = scatter!(
        ax, [Fixtures.test_locations();;
            ball(0.1, 10);;
            ball(0.3, 30);;
            ball(0.6, 60);;
            stack((t) -> starfish(t, 0.9), 0:0.1:2pi)
        ], label="Test Locations", strokewidth=1, color=:red,
        marker=:star4, strokecolor=:black,
    )

# for c in [0., 0.01, 0.05, 0.1, 0.5]
#         visualize!(ax, DiscreteClosedCurve(100, (t) -> starfish(t, 1-c)), false, false)
#     end

    visualize!(ax, curve, false, false)

    # Hide interior axis labels/decorations
    if col > 1
        hideydecorations!(ax, grid=false)
    end
    if row < n_rows
        hidexdecorations!(ax, grid=false)
    end

    axislegend(
        ax,
        [MarkerElement(marker=:circle, color=:transparent) for _ in 1:2],
        [
            "n = $(numpoints(sol))",
            "δ = $(cutoff(k.evalmethod))"
        ],
        position=:lt,       # :lt = left-top (or :rt, :lb, :rb)
        framecolor=:gray50,
        backgroundcolor=(:white, 0.85),
        patchsize=(0, 0)    # hide icon space so only text shows
    )
    # linkaxes!(ax, content(fig[1, 1])...)

    Colorbar(
        fig[row, col][1, 2],
        co;
        # label="log10 error",
        tellheight=false,
        ticks=LinearTicks(10)
    )
end
# Colorbar(fig[:, n_cols+1], co, label="log10 error")


display(fig)

# problem setup plot
# cof = tricontourf!(ax, Γ, x_dense, u_dense, σ;
#     levels=range(extrema(u)..., 10))
# Colorbar(fig[1, 2], cof)
# val = u_dense

const PLOTFILE = if nameof(Makie.current_backend()) === :CairoMakie
    joinpath("figures", FILE * ".pdf")
else
    joinpath("figures", FILE * ".png")
end

save(PLOTFILE, fig)
@info "saved `fig` to $(PLOTFILE)"

fig
