




# for no derivative requested
function map2disc(
    boundary::DiscreteClosedCurve,
    boundary_parameter::AbstractVector, # pass the parameter used, later include this information inside boundary struct
    points::AbstractMatrix,
    relative_cutoff=0.05,
)
    map2disc(nothing, boundary, boundary_parameter, points, relative_cutoff)
end


function map2disc(
    derivative_request::Union{AbstractDerivativeRequest,Nothing},
    boundary::DiscreteClosedCurve,
    boundary_parameter::AbstractVector, # pass the parameter used, later include this information inside boundary struct
    points::AbstractMatrix,
    relative_cutoff=0.05,
)

    laplace=Laplace()
    indirect=Indirect()
    hypersingular_correction=Zeta(32)
    interior=Interior()

    sigma_xi = cos.(boundary_parameter)
    sigma_eta = sin.(boundary_parameter)

    pbs = [
        BoundaryValueProblem(laplace, Dirichlet(sigma_xi), interior, boundary),
        BoundaryValueProblem(laplace, Dirichlet(sigma_eta), interior, boundary),
    ]

    phis = solve.(
        pbs,
        Ref(indirect),
    )

    evals = evaluate.(
        Ref(derivative_request),
        pbs,
        Ref(indirect),
        Ref(hypersingular_correction),
        phis,
        Ref(points),
        Ref(relative_cutoff)
    )

    solns = last.(evals)


    # stack in column-major matrix of 2d point
    xi = stack(first.(solns); dims=1)
    sigma = stack((sigma_xi, sigma_eta); dims=1)
    tau = stack(data.(getindex.(solns, 2)); dims=1)


    if isnothing(derivative_request)
        return xi, sigma, tau
    else
        derivs = stack(first.(evals); dims=2)
        return derivs, xi, sigma, tau
    end
end

