



function map2disc(
    boundary::DiscreteClosedCurve,
    boundary_parameter::AbstractVector, # pass the parameter used, later include this information inside boundary struct
    points::AbstractMatrix,
)

    laplace=Laplace()
    indirect=Indirect()
    hypersingular_correction=Zeta(16)
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

    tau_xi = data(solns[1][2])
    tau_eta = data(solns[2][2])


    # stack in column-major matrix of 2d point
    xi_eta = permutedims(hcat(xi, eta))

    return xi_eta, (sigma_xi, sigma_eta), (tau_xi, tau_eta)
end


function map2disc_with_jacobian(
    boundary::DiscreteClosedCurve,
    boundary_parameter::AbstractVector, # pass the parameter used, later include this information inside boundary struct
    points::AbstractMatrix,
)

    jac = stack(
        map(1:2) do i
            d_points = Enzyme.make_zero(points)
            d_boundary = Enzyme.make_zero(boundary)
            d_boundary_parameter = Enzyme.make_zero(boundary_parameter)
            d_points = Enzyme.make_zero(points)
            d_points[i, :] .= 1.

            ad = autodiff(
                Enzyme.set_runtime_activity(ForwardWithPrimal),
                map2disc,
                Duplicated(boundary, d_boundary),
                Duplicated(boundary_parameter, d_boundary_parameter),
                Duplicated(points, d_points),
            )

            @show typeof(ad[1][1]), size(ad[1][1])
            ad[1][1]
        end
        ;
        dims=2
    )

    return jac


end
