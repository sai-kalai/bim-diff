



function map2disc(
    boundary::DiscreteClosedCurve,
    boundary_parameter::AbstractVector, # pass the parameter used, later include this information inside boundary struct
    points::AbstractMatrix,
)

    indirect=Indirect()
    hypersingular_correction=Sidi()
    interior=Interior()

    pbs = [
        BoundaryValueProblem(laplace, Dirichlet(cos.(boundary_parameter)), interior, boundary),
        BoundaryValueProblem(laplace, Dirichlet(sin.(boundary_parameter)), interior, boundary),
    ]

    phis = solve.(
        pbs,
        Ref(indirect),
    )

    solns = evaluate.(
        pbs,
        Ref(indirect),
        Ref(hypersingular_correction),
        phis,
        Ref(points),
        Ref(0.05)
    )

    # return from evaluate is a tuple of (u, trace)
    xi = solns[1][1]
    eta = solns[2][1]

    # stack in column-major matrix of 2d point
    xi_eta = permutedims(hcat(xi, eta))

    return xi_eta
end


function map2disc_gradient(
    boundary::DiscreteClosedCurve,
    boundary_parameter::AbstractVector, # pass the parameter used, later include this information inside boundary struct
    points::AbstractMatrix,
)

    d_xi, xi = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal

    )

end
