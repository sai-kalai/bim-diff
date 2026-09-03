using BoundaryIntegralEquations
using BoundaryIntegralEquations.DevTools


x_test = Fixtures.test_locations()

res = run_all_simulations(x_test; benchmark=true)

const F = "data/benchmark.jld2"

save_object(F, result)

@info "result = "
display(result)

@info "`result` saved to `$F`"

