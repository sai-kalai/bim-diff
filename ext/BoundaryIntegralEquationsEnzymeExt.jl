
# Extension package for enzyme automatic differentiation
#
module BoundaryIntegralEquationsEnzymeExt

using BoundaryIntegralEquations
using Enzyme


# NOTE: further derivatives are needed for faster optimization algs


# TODO: include matrix_factory
function BoundaryIntegralEquations.evaluate(
    ::WithSpatialDerivativeFwd,
    problem::BoundaryValueProblem,
    approach::Indirect,
    correction::HypersingularCorrection,
    bie_solution::BoundaryDensity,
    target::AbstractMatrix,
    # TODO: turn this into an object
    relative_cutoff=0.05,
)

    # TODO: remove magic constant
    ads = map(1:2) do i
        dproblem = Enzyme.make_zero(problem)

        dbie_solution = Enzyme.make_zero(bie_solution)

        dtarget = Enzyme.make_zero(target)
        dtarget[i, :] .= 1.;

        # TODO: figure out how to compute primal only once
        mode = (i==1 || true) ? ForwardWithPrimal : Forward

        # TODO: achieve without runtime activity
        autodiff(
            set_runtime_activity(mode),
            evaluate,
            Duplicated(problem, dproblem),
            Const(approach),
            Const(correction),
            Duplicated(bie_solution, dbie_solution),
            Duplicated(target, dtarget),
            Const(relative_cutoff),
        )
    end

    # stack components of gradient in 2xN array
    gradient = stack(map(ad -> ad[1][1], ads); dims=1)

    primal = ads[1][2]

    return gradient, primal
end

# TODO: re-do reverse mode for new api
# function BoundaryIntegralEquations.spatial_gradient(
#     mode::Enzyme.ReverseMode,
#     problem::BoundaryValueProblem,
#     target::AbstractMatrix,
#     approach::Approach,
#     correction::Union{SingularCorrection,HypersingularCorrection},
# )
#
#
#     d_problem = Enzyme.make_zero(problem)
#     d_target = Enzyme.make_zero(target)
#
#
#
#     forward, reverse = autodiff_thunk(
#         ReverseSplitWithPrimal,
#         Const{typeof(solve_and_evaluate)},
#         Duplicated,
#         Duplicated{typeof(problem)},
#         Const{typeof(approach)},
#         Const{typeof(correction)},
#         Duplicated{typeof(target)},
#     )
#
#
#     tape, result, shadow_result = forward(
#         Const(solve_and_evaluate),
#         Duplicated(problem, d_problem),
#         Const(approach),
#         Const(correction),
#         Duplicated(target, d_target),
#     )
#
#     shadow_result[1].=1.
#
#     rev = reverse(
#         Const(solve_and_evaluate),
#         Duplicated(problem, d_problem),
#         Const(approach),
#         Const(correction),
#         Duplicated(target, d_target),
#         tape,
#     )
#
#     return d_target
#
# end


# function BoundaryIntegralEquations.map2disc(
#     derivative_request::WithSpatialDerivativeFwd,
#     boundary::DiscreteClosedCurve,
#     boundary_parameter::AbstractVector,
#     points::AbstractMatrix,
# )
#
#     r = map2disc(
#         derivative_request,
#         boundary,
#         boundary_parameter,
#         points,
#     )
#
#     @show r
#
#
# end


end


