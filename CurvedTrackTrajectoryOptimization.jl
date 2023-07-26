module CurvedTrackTrajectoryOptimization

using Ipopt
using Random
using LinearAlgebra
using SparseArrays
using Symbolics 
using GLMakie
using Infiltrator
using ProgressMeter
using GLMakie.GeometryBasics

include("trajectory_functions.jl")
include("generate_trajectory.jl")

end # module HW2
