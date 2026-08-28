
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
using BenchmarkTools
using BoundaryIntegralEquations


include("../fixtures.jl")


@doc raw"""
    SimulationContext

Records what parameters were used to run a particular simulation

"""
struct SolverParameters
    approach_t::Type{<:Approach}
    bdrycond_t::Type{<:BoundaryCondition}
    solution_t::Type{<:NumericalSolution}
    correction::AbstractSingularCorrection
    evalmethod::EvaluationMethod
end

function get_kwargs(k::SolverParameters)
    # each plot should have a script that determines the looks, not one style for all

    marker = k.solution_type <: BVPSolution ? :circle :
             k.solution_type <: BDPSolution ? :rect :
             error()

    linestyle = k.approach_t <: Direct ? :dash :
                k.approach_t <: Indirect ? :dot :
                error()

    color = k.evalmethod isa PotentialTheory ? :red :
            k.evalmethod isa DistancePolicy ? k.evalmethod.cutoff :
            error(k.evalmethod)


    # linewidth = k.evalmethod isa DistancePolicy ? 1.5 + log10(max(k.evalmethod.cutoff, 1e-12)) * -0.3 :
    #             k.evalmethod isa PotentialTheory ? 2 :
    #             k.evalmethod isa CauchyIntegral ? 1 :
    #             error()
    linewidth=1.
    return (;
        marker=marker,
        linestyle=linestyle,
        color=color,
        linewidth=linewidth
    )

end

@doc raw"""
    SolutionMetadata

Contains information about a simulation such as runtime

"""
struct SolutionMetadata
    # initial sketch, maybe include Tryal instance here
    trial::Union{BenchmarkTools.Trial,Nothing}
end

const SolutionWithMetadata = Tuple{
    NumericalSolution,SolutionMetadata
}
function SolutionWithMetadata(sol, md)
    return SolutionWithMetadata((sol, md))
end

@doc raw"""
    ConvergenceResult

Stores the metadata and data associated with a group of simulation runs with
different parameters for several discretization sizes

"""
struct ConvergenceResult{T}
    # metadata: parameters used during the runs
    n_vals::Vector{Int}
    cutoff_vals::Vector{T}
    fd_acc_vals::Vector{Int}
    kr_acc_vals::Vector{Int}

    x::Matrix{T} # x locations in col-major

    u_exact::Vector{T} # exact solution at x points
    neumann_exact::Dict{Int,Vector{T}} # exact neumann data for each n val
    dirichlet_exact::Dict{Int,Vector{T}}

    solutions::Dict{
        SolverParameters,Vector{SolutionWithMetadata}
    } # results of the simulations for several n values, grouped by solver parameters
end

function ConvergenceResult(
    n_vals::Vector{Int},
    cutoff_vals::Vector{T},
    fd_acc_vals::Vector{Int},
    kr_acc_vals::Vector{Int},
    x::Matrix{T},
    u_exact::Vector{T},
) where {T}
    return ConvergenceResult{eltype(T)}(
        n_vals,
        cutoff_vals,
        fd_acc_vals,
        kr_acc_vals,
        x,
        u_exact,
        Dict{Int,Vector{T}}(), # exact neumann and dirichlet data for each n value
        Dict{Int,Vector{T}}(), # exact neumann and dirichlet data for each n value
        Dict{SolverParameters,Vector{SolutionWithMetadata}}(),
    )
end


function plot_errors(
    res::ConvergenceResult,
)

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
        xticks=LinearTicks(5),
    )
    ylims!(ax, (1e-16, 1e+2))

    Colorbar(fig[1, 2], limits=extrema(res.cutoff_vals))

    for (key, sols) in res.solutions

        sort!(sols, by=s -> numpoints(s))

        ns = [numpoints(s) for s in sols]

        errs = key.solution_type <: BVPSolution ? [norm(s.u - res.u_exact, Inf) for s in sols] :
               key.solution_type <: BDPSolution ? begin
            key.bc_type <: Dirichlet ? [norm(s.u - res.neumann_exact[numpoints(s)], Inf) for s in sols] :
            key.bc_type <: Neumann ? [norm(s.u - res.neumann_exact[numpoints(s)], Inf) for s in sols] : 0
        end : 0



        kwargs = get_kwargs(key)

        # distinguish lines that end up being the same
        # lw = first(sols) isa DirichletSolution{<:DomainSide,Indirect,Sidi} ? 3 : 2
        #
        # rt = first(sols) isa DirichletSolution{<:DomainSide,Indirect,Sidi} ? pi/2 : 0.
        # rt = first(sols) isa NumericalSolution{<:DomainSide,Indirect} ? pi/2 : 0.

        ms = 12

        al = 0.6

        scatterlines!(
            ax,
            ns,
            errs,
            # linestyle=st.linestyle,
            # color=st.color,
            # marker=get_marker(:solution),
            # linewidth=lw,
            # strokewidth=1,
            # markersize=ms,
            # alpha=al,
            # rotation=rt
            ;
            colorrange=extrema(res.cutoff_vals),
            strokewidth=1,
            kwargs...
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



    @show "HHH"
    α = 0.1
    lines!(ax,
        res.n_vals, # use last iteration for getting ns
        exp.(-α .* res.n_vals),
        ;
        color=:grey,
        conv_style...
    )

    # Legend(
    #     fig[1, 1],
    #     [
    #         # linestyle -> approach
    #         LineElement(linestyle=get_linestyle(DirichletSolution{DomainSide,Direct})),
    #         LineElement(linestyle=get_linestyle(DirichletSolution{DomainSide,Indirect})),
    #         # color -> bc
    #         MarkerElement(color=get_color(DirichletSolution{DomainSide,Approach,Zeta}), marker=:circle),
    #         MarkerElement(color=get_color(DirichletSolution{DomainSide,Approach,Sidi}), marker=:circle),
    #         MarkerElement(color=get_color(NeumannSolution{DomainSide,Approach}), marker=:circle),
    #         # marker -> solution vs cauchy datum
    #         MarkerElement(color=:black, marker=get_marker(:solution)),
    #         MarkerElement(color=:black, marker=get_marker(:boundary)),
    #
    #         # convergence rates
    #         LineElement(; color=:grey, conv_style...),
    #         # LineElement(; color=:black, conv_style...)
    #     ],
    #     [
    #         # linestyle
    #         "Direct",
    #         "Indirect",
    #         # color
    #         "Dirichlet (FD)",
    #         "Dirichlet (Richardson)",
    #         "Neumann",
    #         # marker
    #         "Solution",
    #         "Boundary Trace",
    #         # convergence line
    #         "O(exp(-$α n))",
    #         # "O(n^-$(order_offset))",
    #     ],
    #     "Legend";
    #     tellwidth=false,
    #     halign=:left,
    #     valign=:bottom
    # )

    fig, ax
end


function add_solutions!(res::ConvergenceResult, correction, evalmethod, sols_with_md...)
    foreach(sols_with_md) do sol_with_md
        (sol, md) = sol_with_md
        push!(
            get!(
                res.solutions,
                SolverParameters(
                    approach(sol.alg),
                    boundary_condition(sol),
                    typeof(sol),
                    correction,
                    evalmethod,
                ),
                Vector{Float64}() # why vector of float...?
            ),
            sol_with_md
        )
    end

end

function Base.show(io::IO, ::MIME"text/plain", res::ConvergenceResult{T}) where {T}
    println(io, "ConvergenceResult{", T, "}:")
    println(io, "  n_vals:          ", res.n_vals)
    println(io, "  cutoff_vals:     ", res.cutoff_vals)
    println(io, "  fd_acc_vals:     ", res.fd_acc_vals)
    println(io, "  kr_acc_vals:     ", res.kr_acc_vals)
    println(io, "  x:               ", summary(res.x))
    println(io, "  u_exact:         ", summary(res.u_exact))
    println(io, "  neumann_exact:   Dict with ", length(res.neumann_exact), " entries")
    println(io, "  dirichlet_exact: Dict with ", length(res.dirichlet_exact), " entries")
    print(io, "  solutions:       ", length(res.solutions), "-element", typeof(res.solutions))
end

@doc raw"""

run all methods with different parameters

"""
function convergence_study(;
    n_vals=20:20:400,
    cutoff_vals=[0.0, 0.01, 0.05, 0.1, 0.5],
    fd_acc_vals=[4, 8, 16, 32],
    kr_acc_vals=fd_acc_vals,
    approach_types=[Direct, Indirect],
    bc_types=[Dirichlet, Neumann],
    viz=false,
    # indicate how to reserve memory
    allocator=(_m, _n) -> Matrix{Float64}(undef, _m, _n),
    benchmark=false,
)
    @show n_vals
    @show cutoff_vals
    @show fd_acc_vals
    @show kr_acc_vals


    # common variables
    laplace = Laplace()
    interior = Interior()
    exterior = Exterior()
    direct = Direct()
    indirect = Indirect()

    sidi = Sidi()

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
    n_dense = 60
    Γ_dense = DiscreteClosedCurve(n_dense, starfish)
    xmin, xmax, ymin, ymax = extrema(Γ_dense)
    xs = range(xmin, xmax, length=n_dense)
    ys = range(ymin, ymax, length=n_dense)
    iter = Iterators.product(xs, ys)
    x_dense = stack(((x, y),) -> SA[x, y], iter; dims=2)

    # get known solution at test and plot points
    x_source, density_source = point_sources()
    density_source = BoundaryDensity(density_source)

    # operators for exact solution at test and plot points
    Γ_source = make_dummy_curve(x_source)
    S_manuf = SingleLayer(laplace, Γ_source, x_test; matrix_factory=allocator, populate_matrix=true)
    S_manuf_dense = SingleLayer(laplace, Γ_source, x_dense; populate_matrix=true)
    u_exact = S_manuf * density_source # exact solution at test points
    u_exact_dense = S_manuf_dense * density_source

    # accumulate results
    res = ConvergenceResult(collect(n_vals), cutoff_vals, fd_acc_vals, kr_acc_vals,
        x_test, u_exact)


    # verify that results match MATLAB version
    u_exact_reference = reference_exact_solution()
    @test u_exact[1:length(u_exact_reference)] ≈ u_exact_reference atol = 1e-15

    # storage for produced solutions

    for n in n_vals

        @show n

        # boundary discretization
        Γ = DiscreteClosedCurve(n, starfish)

        # operators for computing boundary conditions from point sources
        S_source = SingleLayer(laplace, Γ_source, Γ.x)
        D_star_source = AdjointDoubleLayer(laplace, Γ_source, Γ.x)
        populate_matrices!(Γ_source, Γ.x, S_source, D_star_source; target_normals=Γ.n)
        # TODO: test this
        # @assert D_star_source.matrix ≈ AdjointDoubleLayer(laplace, Γ.x, Γ.n, Γ_source; matrix_factory=allocator).matrix
        σ_exact = S_source * density_source # Dirichlet BC
        τ_exact = D_star_source * density_source # Neumann BC exact solution


        res.dirichlet_exact[n] = σ_exact
        res.neumann_exact[n] = τ_exact

        for side in [interior,], bc in [Dirichlet(σ_exact), Neumann(τ_exact)]

            if !any(T -> bc isa T, bc_types)
                @warn "skipping $bc"
                continue
            end

            bvp = BoundaryValueProblem(laplace, bc, side, Γ)

            # NOTE: actually, operators of different orders can be precomputed
            # in parallel using tuple comprenhension
            # ops = (
            #   (SingleLayer(laplace, Γ, KapurRokhlin(x)) for x kr_acc_vals)...,
            #   (Hypersingular(laplace, Γ, Zeta(x)) for x fd_acc_vals)...,
            #   Sidi(),
            #   )
            # populate_matrices!(Γ, ops)
            # for op in ops
            #    solve_and_evaluate(
            #       prob, approach, (op, op isa SingleLayer ? D_star : op isa Hypersingular?  etc...
            #    ) -> requires putting correct args
            # end

            corrections = bc isa Dirichlet ? [Sidi(); [Zeta(x) for x in fd_acc_vals]] :
                          bc isa Neumann ? [KapurRokhlin(x) for x in kr_acc_vals] :
                          error("invalid bc")


            for correction in corrections


                if Direct in approach_types
                    # direct approach

                    u, cauchy_data = solve_and_evaluate(
                        bvp,
                        direct,
                        correction,
                        x_test,
                    )

                    if benchmark
                        trial = @benchmark solve_and_evaluate(
                            $bvp,
                            $direct,
                            $correction,
                            $x_test,
                        )
                    else
                        trial = nothing
                    end

                    # dummy placeholder for now
                    bie_sln = BIESolution(
                        zeros(n),
                        BIEProblem{Direct}(bvp),
                        BIEAlgorithm{Direct}(correction),
                    )

                    bvp_sln = BVPSolution(
                        u,
                        bie_sln,
                        bvp,
                        BVPAlgorithm{Direct}(),
                    )
                    bdp_sln = BDPSolution(
                        data(cauchy_data),
                        bie_sln,
                        BDProblem{Direct}(bvp),
                        BDPAlgorithm{Direct}(),
                    )

                    add_solutions!(res, correction, PotentialTheory(),
                        SolutionWithMetadata(bvp_sln, SolutionMetadata(trial)),
                        SolutionWithMetadata(bdp_sln, SolutionMetadata(trial))
                    )
                end

                if Indirect in approach_types
                    # indirect approach: cutoff is available
                    for cutoff in cutoff_vals
                        if bc isa Neumann
                            continue
                        end

                        if benchmark
                            trial = @benchmark solve_and_evaluate(
                                $bvp,
                                $indirect,
                                $correction,
                                $x_test,
                                $cutoff,
                            )
                        else
                            trial = nothing
                        end
                        u, cauchy_data = solve_and_evaluate(
                            bvp,
                            indirect,
                            correction,
                            x_test,
                            cutoff,
                        )

                        # dummy placeholder
                        bie_sln = BIESolution(
                            zeros(n),
                            BIEProblem{Indirect}(bvp),
                            BIEAlgorithm{Indirect}(),
                        )

                        method = cutoff == 0. ? PotentialTheory() :
                                 isinf(cutoff) ? CauchyIntegral() :
                                 DistancePolicy(cutoff)

                        bvp_sln = BVPSolution(
                            u,
                            bie_sln,
                            bvp,
                            BVPAlgorithm{Indirect}(method),
                        )

                        bdp_sln = BDPSolution(
                            data(cauchy_data),
                            bie_sln,
                            BDProblem{Indirect}(bvp),
                            BDPAlgorithm{Indirect}(correction),
                        )

                        add_solutions!(res, correction, method,
                            SolutionWithMetadata(bvp_sln, SolutionMetadata(trial)),
                            SolutionWithMetadata(bdp_sln, SolutionMetadata(trial))
                        )
                    end
                end
            end
        end
    end

    return res
end



if abspath(PROGRAM_FILE) == @__FILE__
    wait(display(plot_errors(convergence_study())))
end
