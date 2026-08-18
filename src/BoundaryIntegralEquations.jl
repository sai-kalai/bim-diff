module BoundaryIntegralEquations



#
# external packages
#
using LinearAlgebra
using StaticArrays
using FFTA
using LinearSolve
using PolygonOps
using NearestNeighbors

#
# type definitions
#
abstract type IntegralOperator end



# TODO: bvp should already be aware of not only the pde, but also type of bc, side of domain
# solve stage should allow choice of approach
abstract type DifferentialEquation end
struct Laplace <: DifferentialEquation end
struct Helmholtz <: DifferentialEquation end
struct Stokes <: DifferentialEquation end




abstract type HypersingularCorrection end
struct Sidi <: HypersingularCorrection end
struct Zeta <: HypersingularCorrection
    order::Int
end

abstract type SingularCorrection end
struct KapurRokhlin <: SingularCorrection
    order::Int
end


abstract type DomainSide end
struct Interior <: DomainSide end
struct Exterior <: DomainSide end

abstract type Approach end
struct Direct <: Approach end
struct Indirect <: Approach end



@doc raw"""
    AbstractDerivativeRequest

Specify that a derivative needs to be computed, and what mode AD to use

"""
abstract type AbstractDerivativeRequest end
struct WithSpatialDerivativeFwd <: AbstractDerivativeRequest end
struct WithSpatialDerivativeRev <: AbstractDerivativeRequest end

function evaluate(::AbstractDerivativeRequest, args...)
    error("This derivative request requires loading Enzyme")
end




# includes
#
include("finite_differences.jl")
include("kapur_rokhlin_sep_log.jl")
include("densities.jl")
include("manifolds.jl")
include("operators.jl")
include("kernels.jl")
include("solvers.jl")
include("close_evaluation.jl")
include("map2disc.jl")
include("utils.jl")

#
# exports
#
export DiscreteClosedCurve, make_dummy_curve

export DifferentialEquation, Laplace, Helmholtz, Stokes
export HypersingularCorrection, Sidi, Zeta
export SingularCorrection, KapurRokhlin
export DomainSide, Interior, Exterior
export IntegralOperator, SingleLayer, DoubleLayer, AdjointDoubleLayer, Hypersingular
export Approach, Direct, Indirect
export BoundaryCondition, Dirichlet, Neumann
export kernel
export AbstractBoundaryDensity, BoundaryDensity, BoundaryCondition, Dirichlet, Neumann, data
export cauchy_integral, holomorphism_boundary_limit
export AbstractDerivativeRequest, WithSpatialDerivativeFwd, WithSpatialDerivativeRev


export kernel, solution_derivative
export populate_matrices!

export BoundaryValueProblem, solve, evaluate, solve_and_evaluate

export starfish, ball

export map2disc, inverse_map2disc

export visualize

export solution_derivative




function visualize()
    error("No GLMakie detected, please import it before calling")
end


# trick lsp
@static if false
    include("../scripts/main.jl")
    include("../scripts/precomputed_coeffs.jl")

    include("../test/quick_test.jl")
    include("../test/convergence/laplace_2d.jl")
    include("../test/operators.jl")
    include("../test/close_evaluation.jl")
    include("../test/map2disc_ellipse.jl")


end

end # module BoundaryIntegralEquations
