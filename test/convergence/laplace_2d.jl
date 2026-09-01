
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
using GLMakie
using LinearAlgebra
using BenchmarkTools
using BoundaryIntegralEquations

using BoundaryIntegralEquations.DevTools
using BoundaryIntegralEquations.DevTools: Fixtures


@doc raw"""
    SimulationContext

Records what parameters were used to run a particular simulation

"""

function get_kwargs(k::SolverParameters)
    # each plot should have a script that determines the looks, not one style for all

    marker = k.solution_t <: BVPSolution ? :circle :
             k.solution_t <: BDPSolution ? :rect :
             error()

    linestyle = k.approach_t <: Direct ? :dash :
                k.approach_t <: Indirect ? :dot :
                error()

    @show k.evalmethod
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

    for (key::SolverParameters, group::SolutionGroup) in res.solutions

        sort!(group, by=swm -> numpoints(swm[1]))

        sols = solutions(group)

        ns = [numpoints(s) for s in sols]

        errs = key.solution_t <: BVPSolution ? [norm(s.u - res.u_exact, Inf) for s in sols] :
               key.solution_t <: BDPSolution ? begin
            key.bdrycond_t <: Dirichlet ? [norm(s.u - res.neumann_exact[numpoints(s)], Inf) for s in sols] :
            key.bdrycond_t <: Neumann ? [norm(s.u - res.dirichlet_exact[numpoints(s)], Inf) for s in sols] :
            error("invalid bc type $(key.bdrycond_t)")

        end :
               error("invalid solution type $(key.solution_t)")

        if any(isnan, errs)
            @show key
            @show errs
            @show res.u_exact

            for s in sols
                if any(isnan, s.u)
                    @show numpoints(s), extrema(s.u)
                    @show bvp(s)
                end
            end
        end

        @show extrema(errs)

        # kwargs = get_kwargs(key)

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
            # colorrange=extrema(res.cutoff_vals),
            strokewidth=1,
            # kwargs...
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


# 1. run simulations
#
# 2. check that convergence rate is as expected
#



x_test = Fixtures.test_locations()
x_test = [
    x_test;;
    # ball(0.1, 10);;
    # ball(0.3, 30);;
    # ball(0.6, 60);;
    # # avoid  testing close evaluation for gradient
    # stack((t) -> starfish(t, 0.9), 0:0.1:2pi)
]

result = run_all_simulations(
    x_test;
    cutoff_vals=[0.0,],
    fd_acc_vals=[32,]
)


if abspath(PROGRAM_FILE) == @__FILE__
    wait(display(plot_errors(run_all_simulations())))
end
