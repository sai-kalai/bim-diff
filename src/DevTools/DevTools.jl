module DevTools

using BenchmarkTools # keeping as dependency for now, find better way
using StaticArrays

using ..BoundaryIntegralEquations

export ConvergenceResult, SolverParameters, SolutionMetadata,
    SolutionWithMetadata, add_solutions!, SolutionGroup,
    solutions, metadatas, trials, run_all_simulations, Fixtures

include("Fixtures.jl")

@doc raw"""
    SolverParameters

identifies the parameters used for running one solver

"""
struct SolverParameters
    approach_t::Type{<:Approach}
    bdrycond_t::Type{<:BoundaryCondition}
    solution_t::Type{<:NumericalSolution}
    correction::AbstractSingularCorrection
    evalmethod::EvaluationMethod
end
# function Base.show(io::IO, ::MIME"text/plain", param::SolverParameters)
#     println(io, "SolverParameters:")
#     println(io, "  n_vals:          ", res.n_vals)
#     println(io, "  cutoff_vals:     ", res.cutoff_vals)
#     println(io, "  fd_acc_vals:     ", res.fd_acc_vals)
#     println(io, "  kr_acc_vals:     ", res.kr_acc_vals)
#     println(io, "  x:               ", summary(res.x))
#     println(io, "  u_exact:         ", summary(res.u_exact))
#     println(io, "  neumann_exact:   Dict with ", length(res.neumann_exact), " entries")
#     println(io, "  dirichlet_exact: Dict with ", length(res.dirichlet_exact), " entries")
#     print(io, "  solutions:       ", length(res.solutions), "-element", typeof(res.solutions))
# end

@doc raw"""
    SolutionMetadata

Contains information about a simulation such as runtime

"""
struct SolutionMetadata
    # initial sketch, maybe include Tryal instance here
    trial::Union{BenchmarkTools.Trial,Nothing}
end
const SolutionWithMetadata = Tuple{NumericalSolution,SolutionMetadata}
SolutionWithMetadata(s, m) = SolutionWithMetadata((s, m))
const SolutionGroup = Vector{SolutionWithMetadata}
# iterators for broadcasting
solutions(g::SolutionGroup) = (s for (s, _) in g)
metadatas(g::SolutionGroup) = (m for (_, m) in g)
trials(g::SolutionGroup) = (m.trial for (_, m) in g)

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

    # results of the simulations for several n values, grouped by solver parameters
    solutions::Dict{SolverParameters,SolutionGroup}
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
    manufactured_solution()

computes known solution at given points and useful information at point sources

Returns tuple of:
- source dummy curve
- density at sources
- exact solution at test points

"""
function manufactured_solution(eqn::DifferentialEquation, x_test,)
    # TODO:
    x_source, density_source = Fixtures.point_sources()
    density_source = BoundaryDensity(density_source)

    # operators for exact solution at test and plot points
    Γ_source = make_dummy_curve(x_source)
    S_source = SingleLayer(eqn, Γ_source, x_test, populate_matrix=true)
    u_exact = S_source * density_source # exact solution at test points
    return Γ_source, BoundaryDensity(density_source), u_exact
end

@doc raw"""

run all methods with different parameters

"""
function run_all_simulations(
    x_test::AbstractMatrix, # test locations
    ;
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


    # evaluation points for convergence results
    # x_test = test_locations()
    # x_test = [
    #     x_test;;
    #     ball(0.1, 10);;
    #     ball(0.3, 30);;
    #     ball(0.6, 60);;
    #     # avoid  testing close evaluation for gradient
    #     stack((t) -> starfish(t, 0.9), 0:0.1:2pi)
    # ]


    # dense grid for plotting
    n_dense = 60
    Γ_dense = DiscreteClosedCurve(n_dense, starfish)
    xmin, xmax, ymin, ymax = extrema(Γ_dense)
    xs = range(xmin, xmax, length=n_dense)
    ys = range(ymin, ymax, length=n_dense)
    iter = Iterators.product(xs, ys)
    x_dense = stack(((x, y),) -> SA[x, y], iter; dims=2)


    # get known solution at test and plot points
    # x_source, density_source = point_sources()
    # density_source = BoundaryDensity(density_source)
    # # operators for exact solution at test and plot points
    # Γ_source = make_dummy_curve(x_source)
    # S_manuf = SingleLayer(laplace, Γ_source, x_test; matrix_factory=allocator, populate_matrix=true)
    # u_exact = S_manuf * density_source # exact solution at test points

    Γ_source, density_source, u_exact = manufactured_solution(laplace, x_test)
    # accumulate results
    res = ConvergenceResult(collect(n_vals), cutoff_vals, fd_acc_vals, kr_acc_vals,
        x_test, u_exact)

    # verify that results match MATLAB version
    u_exact_reference = Fixtures.reference_exact_solution()
    # move this to test module
    # @test u_exact[1:length(u_exact_reference)] ≈ u_exact_reference atol = 1e-15

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

                        method = cutoff == 0. ? PotentialTheory() :
                                 isinf(cutoff) ? CauchyIntegral() :
                                 DistancePolicy(cutoff)


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


end
