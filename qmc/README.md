Code has been tested using Julia Versions 1.5.x, 1.6.x

```julia
# Required packages
using MPI
using LinearAlgebra
using Random
using Statistics
using Printf
using CSV
using HDF5
using Dates

# Structure
"""
--- main.jl
    --- initialize.jl
        --- lattices.jl
        --- utilities.jl
    --- montecarlo.jl 
        --- utilities.jl
    --- measure.jl
    --- postprocess.jl
"""