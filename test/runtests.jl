using SafeTestsets


@safetestset "Integral Operators" include("operators.jl")

@safetestset "Close Evaluation" include("close_evaluation.jl")

@safetestset "Map2Disc Ellipse" include("map2disc_ellipse.jl")
