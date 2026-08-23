
# TEST_LAP2D_HYPER_BIE
# Test hypersingular zeta-corrected trapezoidal rule for Laplace layer
# potentials on smooth geometries by solving the BVPs using a direct
# approach to BIE:
#      	int Laplace ansatz: u = S*(du/dn) - D*u
#       int Calder?n projection:     u =  (1/2-D)*u  +         S*(du/dn)
#                                du/dn =       -T*u  + (1/2+D^*)*(du/dn)
#      	ext Laplace ansatz: u = D*u - S*(du/dn) + omega
#       ext Calder?n projection:     u =  (1/2+D)*u  -         S*(du/dn)
#                                du/dn =        T*u  + (1/2-D^*)*(du/dn)
#
# c.f. Hsiao-Wendland 2008, Sec.1.3-1.4

using Test
using Revise
using StaticArrays

using CairoMakie


using LinearAlgebra

using BoundaryIntegralEquations


include("../fixtures.jl")

abstract type Solution end
abstract type NumericalSolution{S,A} end



mutable struct DirichletSolution{S<:DomainSide,A<:Approach,C<:HypersingularCorrection} <: NumericalSolution{S,A}
    n
    u::Vector{Float64}
    τ::Vector{Float64}
    correction::C
    u_err
    τ_err
end

struct NeumannSolution{S<:DomainSide,A<:Approach} <: NumericalSolution{S,A}
    n
    u::Vector{Float64}
    σ::Vector{Float64}
    u_err
    σ_err
end

struct ExactSolution{S<:DomainSide} <: Solution
    n
    u::Vector{Float64}
    σ::Vector{Float64}
    τ::Vector{Float64}
end


function get_trace_err(s::DirichletSolution)
    return s.τ_err
end
function get_trace_err(s::NeumannSolution)
    return s.σ_err
end

solution_label(sol) = begin
    bc =
        sol isa DirichletSolution ? "Dirichlet" :
        sol isa NeumannSolution ? "Neumann" :
        "Unknown"

    approach =
        nameof(typeof(sol).parameters[2])

    correction =
        sol isa DirichletSolution ?
        string(nameof(typeof(sol).parameters[3])) :
        ""

    isempty(correction) ?
    "$bc / $approach" :
    "$bc / $approach / $correction"
end

get_color(::Type{DirichletSolution{S,A,Sidi}}) where {S<:DomainSide,A<:Approach} = :blue
get_color(::Type{DirichletSolution{S,A,Zeta}}) where {S<:DomainSide,A<:Approach} = :red
get_color(::Type{NeumannSolution{S,A}}) where {S<:DomainSide,A<:Approach} = :lawngreen
get_color(s::NumericalSolution) = get_color(typeof(s))
get_linestyle(::Type{<:NumericalSolution{S,Direct}}) where {S<:DomainSide} = :dot
get_linestyle(::Type{<:NumericalSolution{S,Indirect}}) where {S<:DomainSide} = :dash
get_linestyle(s::NumericalSolution) = get_linestyle(typeof(s))
get_marker(data) = begin
    if data == :solution
        return :utriangle
    elseif data == :boundary
        return :rect
    end
end

function solution_style(sol)

    color = get_color(sol)
    linestyle = get_linestyle(sol)

    return (; linestyle, color,)
end


function plot_errors(
    solutions::Vector{NumericalSolution},
)
    # Group by configuration
    groups = Dict{String,Vector{NumericalSolution}}()

    for sol in solutions

        key = solution_label(sol)

        if !haskey(groups, key)
            groups[key] = NumericalSolution[]
        end

        push!(groups[key], sol)
    end

    # Plot
    fig = Figure(
    # size=(900, 600)
    )

    ax = Axis(
        fig[1, 1],
        xlabel="n",
        ylabel="L∞-error",
        yscale=log10,
        xscale=log10,
        xticks=LinearTicks(5)
    )

    ns = nothing
    for (label, sols) in groups

        sort!(sols, by=s -> s.n)

        ns = [s.n for s in sols]

        u_errs = [s.u_err for s in sols]

        st = solution_style(first(sols))


        # distinguish lines that end up being the same
        lw = first(sols) isa DirichletSolution{<:DomainSide,Indirect,Sidi} ? 3 : 2

        rt = first(sols) isa DirichletSolution{<:DomainSide,Indirect,Sidi} ? pi/2 : 0.
        rt = first(sols) isa NumericalSolution{<:DomainSide,Indirect} ? pi/2 : 0.

        ms = 12

        al = 0.6

        scatterlines!(
            ax,
            ns,
            u_errs,
            label=label,
            linestyle=st.linestyle,
            color=st.color,
            marker=get_marker(:solution),
            linewidth=lw,
            strokewidth=1,
            markersize=ms,
            alpha=al,
            rotation=rt
        )

        trace_errs = [
            get_trace_err(s)
            for s in sols
        ]

        scatterlines!(
            ax,
            ns,
            trace_errs,
            label="$label trace",
            linestyle=st.linestyle,
            color=st.color,
            marker=get_marker(:boundary),
            linewidth=2,
            strokewidth=1,
            markersize=ms,
            alpha=al,
        )

    end




    # # trendlines


    conv_style = (; linestyle=:dashdotdot, linewidth=3)

    # polynomial
    order_offset = 32/8

    # lines!(
    #     ax,
    #     ns,
    #     (ns ./ (ns[1])) .^ float(-order_offset),
    #     ;
    #     color=:black,
    #     conv_style...
    # )

    α = 0.1
    lines!(ax,
        ns, # use last iteration for getting ns
        exp.(-α .* ns),
        ;
        color=:grey,
        conv_style...
    )

    Legend(
        fig[1, 1],
        [
            # linestyle -> approach
            LineElement(linestyle=get_linestyle(DirichletSolution{DomainSide,Direct})),
            LineElement(linestyle=get_linestyle(DirichletSolution{DomainSide,Indirect})),
            # color -> bc
            MarkerElement(color=get_color(DirichletSolution{DomainSide,Approach,Zeta}), marker=:circle),
            MarkerElement(color=get_color(DirichletSolution{DomainSide,Approach,Sidi}), marker=:circle),
            MarkerElement(color=get_color(NeumannSolution{DomainSide,Approach}), marker=:circle),
            # marker -> solution vs cauchy datum
            MarkerElement(color=:black, marker=get_marker(:solution)),
            MarkerElement(color=:black, marker=get_marker(:boundary)),

            # convergence rates
            LineElement(; color=:grey, conv_style...),
            # LineElement(; color=:black, conv_style...)
        ],
        [
            # linestyle
            "Direct",
            "Indirect",
            # color
            "Dirichlet (FD)",
            "Dirichlet (Richardson)",
            "Neumann",
            # marker
            "Solution",
            "Boundary Trace",
            # convergence line
            "O(exp(-$α n))",
            # "O(n^-$(order_offset))",
        ],
        "Legend";
        tellwidth=false,
        halign=:left,
        valign=:bottom
    )

    fig, ax
end



function convergence_study(n_vals=20:20:200, accuracy_order=32; viz=false)

    @show n_vals
    cutoff = 0.05
    @show cutoff


    # useful constants
    laplace = Laplace()
    interior = Interior()
    exterior = Exterior()
    kapur_rokhlin = KapurRokhlin(accuracy_order)
    zeta = Zeta(accuracy_order)
    sidi = Sidi()
    direct = Direct()
    indirect = Indirect()

    # indicate how to reserve memory
    allocator = (_m, _n) -> Matrix{Float64}(undef, _m, _n)

    # evaluation points for convergence results
    x_test = test_locations()
    x_test = [
        x_test;;
        ball(0.1, 10);;
        ball(0.3, 30);;
        ball(0.6, 60);;
        # avoid  testing close evaluation for gradient
        stack((t) -> starfish(t, 0.9), 0:0.1:2pi)
    ]

    # dense grid for plotting
    n_dense = 200
    Γ_dense = DiscreteClosedCurve(n_dense, starfish)
    xmin, xmax, ymin, ymax = extrema(Γ_dense)
    xs = range(xmin, xmax, length=n_dense)
    ys = range(ymin, ymax, length=n_dense)
    iter = Iterators.product(xs, ys)
    x_dense = stack(((x, y),) -> SA[x, y], iter; dims=2)

    # get known solution at test and plot points
    x_source, density_source = point_sources()

    ds2 = similar(density_source)

    for i in eachindex(ds2)
        ds2[i] = density_source[mod1(i+5, length(ds2))]
    end

    density_source .= ds2

    density_source = BoundaryDensity(density_source)


    Γ_source = make_dummy_curve(x_source)
    S_manuf = SingleLayer(laplace, Γ_source, x_test; matrix_factory=allocator, populate_matrix=true)
    S_manuf_dense = SingleLayer(laplace, Γ_source, x_dense; populate_matrix=true)
    u_exact = S_manuf * density_source # exact solution at test points
    u_exact_dense = S_manuf_dense * density_source

    # verify that results match MATLAB version
    u_exact_reference = reference_exact_solution()
    # @test u_exact[1:length(u_exact_reference)] ≈ u_exact_reference atol = 1e-15

    # scatter!(ax, x_test[:, 1], x_test[:, 2], color=u_exact)

    # wait(display(fig))

    # println("Printing max-norm errors")
    # println("Interior")

    num_solutions = Vector{NumericalSolution}()

    for (i, n) ∈ enumerate(n_vals)

        Γ = DiscreteClosedCurve(n, starfish) # boundary of the domain


        # target: domain boundary, source: manufactured solution point sources
        S_source = SingleLayer(laplace, Γ_source, Γ.x) # ok
        D_star_source = AdjointDoubleLayer(laplace, Γ_source, Γ.x) # ok
        populate_matrices!(Γ_source, Γ.x, S_source, D_star_source; target_normals=Γ.n)


        # TODO: test this
        # @assert D_star_source.matrix ≈ AdjointDoubleLayer(laplace, Γ.x, Γ.n, Γ_source; matrix_factory=allocator).matrix

        σ = S_source * density_source # Dirichlet BC
        τ_exact = D_star_source * density_source # Neumann BC exact solution


        S = SingleLayer(laplace, Γ, kapur_rokhlin,) # ok
        D = DoubleLayer(laplace, Γ,) # ok
        D_star = AdjointDoubleLayer(laplace, Γ,)  # ok
        H_zeta = Hypersingular(laplace, Γ, zeta,) # ok
        H_sidi = Hypersingular(laplace, Γ, sidi,) # ok
        populate_matrices!(Γ, S, D, D_star, H_sidi, H_zeta)

        S_target = SingleLayer(laplace, Γ, x_test) # ok
        D_target = DoubleLayer(laplace, Γ, x_test) # ok
        populate_matrices!(Γ, x_test, S_target, D_target)


        # Dirichlet Zeta Direct

        u, τ = solve_and_evaluate(
            BoundaryValueProblem(
                laplace,
                Dirichlet(σ),
                interior,
                Γ
            ),
            direct,
            D_star,
            H_zeta,
            S_target,
            D_target,
        )

        if viz
            u_dense, τ_dense = solve_and_evaluate(
                BoundaryValueProblem(
                    laplace,
                    Dirichlet(σ),
                    interior,
                    Γ
                ),
                indirect,
                zeta,
                x_dense,
                cutoff,
            )

            fig, ax = visualize(Γ, false, false)

            # cof = tricontourf!(ax, Γ, x_dense, u_dense, σ;
            #     levels=range(extrema(u)..., 10))
            # Colorbar(fig[1, 2], cof)
            # val = u_dense

            val = log10.(abs.(u_dense - u_exact_dense) .+ eps(eltype(u_dense)))

            outside_mask = mask(Γ, x_dense, Exterior())

            # val[outside_mask] .= NaN
            val = reshape(val, (n_dense, n_dense))

            lo, hi = extrema(val[.! outside_mask])

            step = (hi-lo) < 5 ? 0.5 : 1

            levels = range(floor(lo), ceil(hi), step=step)

            co = contourf!(
                ax,
                Γ,
                xs,
                ys,
                val,
                levels=levels,
                extendlow=:auto,
                extendhigh=:auto,
            )


            Colorbar(
                fig[1, 2],
                co;
                label="log10 error",
                ticks=levels
            )

            return fig, ax
        end


        push!(
            num_solutions,
            DirichletSolution{Interior,Direct,Zeta}(
                n,
                u,
                τ,
                zeta,
                norm(u_exact - u, Inf),
                norm(τ_exact - τ, Inf)
            )
        )

        # Dirichlet Zeta Indirect
        u, τ = solve_and_evaluate(
            BoundaryValueProblem(
                laplace,
                Dirichlet(σ),
                interior,
                Γ
            ),
            indirect,
            zeta,
            x_test,
            cutoff,
        )
        push!(
            num_solutions,
            DirichletSolution{Interior,Indirect,Zeta}(
                n,
                u,
                τ,
                zeta,
                norm(u_exact - u, Inf),
                norm(τ_exact - τ, Inf)
            )
        )

        # Dirichlet Sidi Direct
        # hypersingular operator using Sidi's staggered grid
        u, τ = solve_and_evaluate(
            BoundaryValueProblem(
                laplace,
                Dirichlet(σ),
                interior,
                Γ
            ),
            direct,
            D_star,
            H_sidi,
            S_target,
            D_target,
        )
        push!(
            num_solutions,
            DirichletSolution{Interior,Direct,Sidi}(
                n,
                u,
                τ,
                sidi,
                norm(u_exact - u, Inf),
                norm(τ_exact - τ, Inf)
            )
        )
        # Dirichlet Sidi Indirect
        u, τ = solve_and_evaluate(
            BoundaryValueProblem(
                laplace,
                Dirichlet(σ),
                interior,
                Γ
            ),
            indirect,
            sidi,
            x_test,
            cutoff,
        )
        push!(
            num_solutions,
            DirichletSolution{Interior,Indirect,Sidi}(
                n,
                u,
                τ,
                sidi,
                norm(u_exact - u, Inf),
                norm(τ_exact - τ, Inf)
            )
        )


        # Neumann problem
        # swap bdry conditions
        σ_exact = σ
        τ = τ_exact

        u, σ = solve_and_evaluate(
            BoundaryValueProblem(
                laplace,
                Neumann(τ),
                interior,
                Γ
            ),
            direct,
            S,
            D,
            S_target,
            D_target,
        )
        # "recover constant" in the original code
        offset = u_exact[1] - u[1]
        u .+= offset
        data(σ) .+= offset # TODO: put this inside solver maybe and user passes integration constant
        push!(
            num_solutions,
            NeumannSolution{Interior,Direct}(
                n,
                u,
                σ,
                norm(u_exact - u, Inf),
                norm(σ_exact - σ, Inf))
        )

        u, σ = solve_and_evaluate(
            BoundaryValueProblem(
                laplace,
                Neumann(τ),
                interior,
                Γ
            ),
            indirect,
            S,
            D_star,
            S_target,
        )
        # "recover constant" in the original code...
        offset = u_exact[1] - u[1]
        u .+= offset
        data(σ) .+= offset # TODO: put this inside solver maybe and user passes integration constant
        push!(
            num_solutions,
            NeumannSolution{Interior,Indirect}(
                n,
                u,
                σ,
                norm(u_exact - u, Inf),
                norm(σ_exact - σ, Inf)
            )
        )


    end

    return num_solutions


end



if abspath(PROGRAM_FILE) == @__FILE__
    wait(display(plot_errors(convergence_study())))
end
