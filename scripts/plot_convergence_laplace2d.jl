# Plot the convergence behavior using Potential Theory
#
# First acquire the convergence data by running the script from the project's
# root directory
#
#```bash
#$ julia --project=. test/convergence/laplace_2d.jl
#```
#
#or from the REPL
#
#```julia
#julia> include("test/convergence/laplace_2d.jl")
#```
#
using JLD2
using GLMakie

using BoundaryIntegralEquations, BoundaryIntegralEquations.DevTools

include("plot_utils.jl")


const FILE = "convergence_laplace_2d"
const DATAFILE = joinpath("data", FILE * ".jld2")
const PLOTFILE = if nameof(Makie.current_backend()) === :CairoMakie
    joinpath("figures", FILE * ".pdf")
else
    joinpath("figures", FILE * ".png")
end

result = load_object(DATAFILE)
@info "loaded `result` from $DATAFILE"

fig, ax = plot_errors(result)

save(PLOTFILE, fig)
@info "saved `fig` to $(PLOTFILE)"

fig
