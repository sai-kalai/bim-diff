
using Test
using Revise
using LinearAlgebra
using StaticArrays
using Statistics
using PolygonOps
using GLMakie
using Enzyme



using BoundaryIntegralEquations


function main(viz=false)

    γ = deg2rad(30.)
    x0 = 2.
    y0 = 3.
    a = 5.
    b = 2.


    function rotated_ellipse(θ, x0, y0, a, b, γ)

        mat = SA[
            cos(θ)*cos(γ) -sin(θ)*sin(γ)
            cos(θ)*sin(γ) sin(θ)*cos(γ)
        ]
        SA[x0, y0] + mat * SA[a, b]
    end

    """
        exact_solution(xy)

    compute the xi, eta values for given pair of xy values

    # Arguments
    - `xy`: two-element array containing the x and y coordinates
    """
    function exact_solution(xy, x0, y0, a, b, γ)
        x, y = xy[1], xy[2]
        xi = (x - x0) * cos(γ) + (y - y0) * sin(γ)
        eta = (x - x0) * -sin(γ) + (y - y0) * cos(γ)
        return SA[xi/a, eta/b]
    end


    n_grid = 30
    θ_grid = range(0, 2π; length=n_grid + 1)[1:(end-1)]

    boundary_grid = stack((t) -> rotated_ellipse(t, x0, y0, a, b, γ), θ_grid; dims=2)

    xmin, ymin = minimum(boundary_grid, dims=2) |> vec
    xmax, ymax = maximum(boundary_grid, dims=2) |> vec

    xs = range(xmin, xmax, length=n_grid)
    ys = range(ymin, ymax, length=n_grid)


    x_test_all = stack(((x, y),) -> SA[x, y], Iterators.product(xs, ys); dims=2)

    xi_eta_exact_all = stack(
        (t) -> exact_solution(t, x0, y0, a, b, γ),
        eachcol(x_test_all);
        dims=2
    )
    xi_eta_exact_boundary = stack(
        (t) -> exact_solution(t, x0, y0, a, b, γ), eachcol(boundary_grid);
        dims=2
    )


    n_vals = 200:80:400
    errs = zeros(Float64, size(n_vals, 1))

    ord = 32

    laplace = Laplace()
    interior = Interior()
    zeta = Zeta(ord)
    sidi = Sidi()
    indirect = Indirect()


    for (i, n) in enumerate(n_vals)

        # Define boundary
        θ = range(0, 2π; length=n + 1)[1:(end-1)]
        Γ = DiscreteClosedCurve(θ, (t) -> rotated_ellipse(t, x0, y0, a, b, γ))


        # check which points are inside

        poly = [[col[1], col[2]] for col in eachcol(Γ.x)]
        push!(poly, Γ.x[:, 1])

        mask = [
            inpolygon((x_test_all[1, col], x_test_all[2, col]), poly) == 1
            for col in axes(x_test_all, 2)
        ]
        x_test = x_test_all[:, mask]
        xi_exact = xi_eta_exact_all[:, mask]

        relative_cutoff = 0.05

        xi_num, sigma, tau = map2disc(Γ, θ, x_test, relative_cutoff)

        jac, xi_num_2, sigma_2, tau_2 = map2disc(
            WithSpatialDerivativeFwd(), Γ, θ, x_test, relative_cutoff)

        @test xi_num ≈ xi_num_2
        @test sigma ≈ sigma_2
        @test tau ≈ tau_2

        # NOTE: expected is actually wrong close to the boundary, need to
        # implement Cauchy integral for solution gradient
        jac_expected = stack(
            map(1:2) do i
                solution_derivative(
                    Direct(),
                    x_test,
                    Γ.x,
                    Γ.n,
                    Γ.w,
                    tau[i, :],
                    sigma[i, :],
                )

            end
            ;
            dims=2
        )

        jac_err = jac - jac_expected

        # frobenius norm
        jac_err_norm = vec(sqrt.(sum(abs2, jac_err; dims=(1, 2))))


        e = xi_num .- xi_exact

        # Nx1, store euclidean norm of error for each point
        e_norm = norm.(eachcol(e), 2) .+ eps(Float64)

        # 1x1
        errs[i] = maximum(e_norm)

        @show n, errs[i], median(jac_err_norm)


        # test inverse mapping
        xi = reshape([0., 0.], (2, 1)) # client wants to find x corresponding to this xi
        x0 = reshape([x0, y0] .+ randn(2), (2, 1)) # initial guess for x
        println("searching for xi=$xi")
        println("initial x = $x0")
        x = inverse_map2disc(Γ, θ, xi, relative_cutoff, x0)

        @show x
        # quadrature nodes

        n_quadrature = 5
        rho_nodes = range(0, 1; length=n_quadrature + 1)[2:(end-1)]
        theta_nodes = range(0, 2π; length=n_quadrature + 1)[1:(end-1)]

        # list containing all integration points
        xi_nodes = stack(
            ((rho, theta),) -> SA[rho*cos(theta), rho*sin(theta)],
            Iterators.product(rho_nodes, theta_nodes),
            ;
            dims=2
        )

        x_nodes = inverse_map2disc.(
            Ref(Γ),
            Ref(θ),
            eachcol(xi_nodes),
            Ref(relative_cutoff),
            Ref(x0),
        )

        # jac_nodes, xi_nodes, _ = map2disc(WithSpatialDerivativeFwd(), Γ, θ, x_nodes, relative_cutoff)
        #
        # # shape optimization functional using trapezoidal rule
        # functional = 0.0
        # for i in axes(jac_nodes, 3)
        #     functional += 1/2*det(jac[:, :, i])
        # end
        #
        # functional *= (rho_nodes[2] - rho_nodes[1]) * (theta_nodes[2] - theta_nodes[1])




        if viz
            fig, ax = visualize(Γ)
            scatter_kwargs = (;
                colorscale=log10,
                color=jac_err_norm,
                markersize=15,
                colormap=:viridis,
            )
            arrow_kwargs = (;
                lengthscale=1.)
            arrows = jac_err[1, :, :]
            sc1 = scatter!(ax, x_test[1, :], x_test[2, :]; scatter_kwargs...)



            arr = arrows2d!(ax, x_test[1, :], x_test[2, :], arrows[1, :], arrows[2, :]; arrow_kwargs...)
            ax2 = Axis(fig[1, 2]; aspect=DataAspect(), title="n = $n")
            lines!(ax2, xi_eta_exact_boundary[1, :], xi_eta_exact_boundary[2, :]; color=:black)
            sc2 = scatter!(ax2, xi_exact[1, :], xi_exact[2, :]; scatter_kwargs...)
            scatter!(ax2, xi_nodes[1, :], xi_nodes[2, :]; color=:red)
            Colorbar(fig[1, 2][1, 3], sc2; label="log10 error inf norm")
            wait(display(fig))
        end


        break



    end

    # convergence plot
    # fig3 = Figure()
    # ax3 = Axis(fig3[1, 1]; xscale=log10, yscale=log10)
    #
    # scatterlines!(ax3, n_vals, errs)
    # lines!(ax3, n_vals, (n_vals ./ n_vals[1]) .^ (-2), label="O(h^-2)")
    # lines!(ax3, n_vals, (n_vals ./ n_vals[1]) .^ (-1), label="O(h^-1)")
    # axislegend(ax3)

    # wait(display(fig3))

end



if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
