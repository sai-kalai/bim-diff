# Plot the solution efficiency = accuracy / time
# To acquire the data, first run `benchmark/benchmark.jl`

using BoundaryIntegralEquations, BoundaryIntegralEquations.DevTools
using Statistics
using GLMakie
using JLD2

include("plot_utils.jl")

const FILE = "benchmark"
const DATAFILE = joinpath("data", FILE * ".jld2")

res = load_object(DATAFILE)

@info "loaded `res` from $DATAFILE"


fig = Figure()
ax = Axis(
    fig[1, 1],
    xlabel="n",
    ylabel="efficiency",
    xscale=log10,
    yscale=log10,
)

# filter
pred(x) = 16<=x<=32
filter!(pred, res.kr_acc_vals)
filter!(pred, res.fd_acc_vals)
for (key, group) in filter(((k, v),) -> begin
        # if k.approach_t <: Indirect
        #     return false
        # end
        if k.solution_t <: BDPSolution
            return false
        end
        # if k.bdrycond_t <: Neumann
        #     return false
        # end
        if k.correction isa Union{Zeta,KapurRokhlin}
            return pred(k.correction.order)
        end
        return true

    end, res.solutions)

    sols = solutions(group)
    ns = numpoints.(sols)

    errs = errors(key, res, group)

    # compute efficiencies
    mids = errs ./ median.(times(group))
    los = errs ./ quantile.(times(group), 0.4)
    his = errs ./ quantile.(times(group), 0.6)

    kwargs = scatterlines_common_kwargs(key, res)

    scatterlines!(ax, ns, mids; kwargs...)

    rangebars!(
        ax, ns, los, his,
        whiskerwidth=10,
        colormap=kwargs.markercolormap,
        colorrange=kwargs.markercolorrange,
        color=fill(kwargs.markercolor, length(ns)),
    )


end


c1, c2 = scatterlines_common_colorbars!(fig, res)
legend = scatterlines_common_legend!(fig, res)
legend.halign=:left
fig[1, 1] = legend

const PLOTFILE = if nameof(Makie.current_backend()) === :CairoMakie
    joinpath("figures", FILE * ".pdf")
else
    joinpath("figures", FILE * ".png")
end
save(PLOTFILE, fig)
@info "saved `fig` to $(PLOTFILE)"
fig


