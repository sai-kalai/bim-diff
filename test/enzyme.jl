using BoundaryIntegralEquations
using Test

@testset "Enzyme Autodiff" begin

    n = 20
    ord = 32

    Γ = DiscreteClosedCurve(n, starfish)

    x_test = test_locations()
    @show size(x_test)

    laplace = Laplace()
    kapur_rokhlin = KapurRokhlin(ord)
    zeta = Zeta(ord)
    sidi = Sidi()


    allocator = (_m, _n) -> Matrix{Float64}(undef, _m, _n)

    @testset "Solution gradient" begin


    end
end

