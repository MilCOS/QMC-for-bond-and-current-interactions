Code has been tested using Julia Versions 1.5.x, 1.6.x, 1.7.x

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
--- system.csv
--- main.jl
    --- initialize.jl
        --- lattices.jl
        --- utilities.jl
    --- montecarlo.jl 
        --- utilities.jl
    --- measure.jl
    --- postprocess.jl
"""
```

To run the code
```sh
mpiexec -n 8 julia ./main.jl
```
