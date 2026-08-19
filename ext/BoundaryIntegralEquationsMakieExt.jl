
module BoundaryIntegralEquationsMakieExt

using BoundaryIntegralEquations
using Makie



function BoundaryIntegralEquations.visualize(
    b::DiscreteClosedCurve,
    arrows::Bool=true,
)

    fig = Figure()
    ax = Axis(fig[1, 1]; aspect=DataAspect())


    x, y = eachrow(b.x)

    curve = lines!(ax, x, y; color=1:size(b, 2))

    if arrows
        norm = arrows2d!(ax, x, y, b.n[1, :], b.n[2, :],
            color="red",
            lengthscale=0.1,
        )
        tang = arrows2d!(ax, x, y, real.(b.cw), imag.(b.cw),
            color="red",
            lengthscale=0.1,
        )
    end

    # Legend(fig[1, 1][1, 2],
    #     [curve, norm],
    #     ["curve", "normal"];
    #     # tellwidth=false,
    #     # halign=:left,
    #     # valign=:bottom,
    # )

    return fig, ax

end


function BoundaryIntegralEquations.visualize(
    b::DiscreteClosedCurve,
    positions::AbstractMatrix,
    values::AbstractVector,
    ;
    kwargs...
)

    fig, ax = BoundaryIntegralEquations.visualize(b, false)

    # extract from col-major ordering
    x, y = eachrow(positions)

    Makie.tricontourf!(ax, x, y, values; kwargs...)

    Makie.scatter!(ax, x, y; color=:red, marker=:cross)

    return fig, ax
end

end
