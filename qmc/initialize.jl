# Copyright (c) 2022, Han Xu
# All rights reserved.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree. 


const _mattype = ComplexF64
const comm = MPI.COMM_WORLD
const ncpurank = MPI.Comm_size(comm)
const icpurank = MPI.Comm_rank(comm)
const rootcpurank = 0


include("lattices.jl")
include("utilities.jl")


parms = CSV.File("system.csv", delim=" ", comment="#", transpose=true)
# system
const Row = parms.row[1]
const Col = parms.col[1]
const N = parms.n[1]
const Ns = parms.ns[1]
const U = [parms.gsbd[1], parms.gtcur[1]]
const tnn = parms.tnn[1]
const μ = parms.mu[1] # given mu ≠ real mu
const doping = parms.doping[1] # given doping should equal to the real doping. For example, 1/16, 1/8, 1/4, 1/2
const tryflux = parms.tryflux[1] # 0.0001
# Monte Carlo basic
const WarmStep = parms.warmstep[1]
const MeasureStep = parms.measurestep[1]
const seed = parms.seed[1]
# QMC basic
const β = parms.beta[1]
const Interval = parms.interval[1]
const P0 = parms.p0[1]
const Δτ = β / Interval
const Projection = round(Integer, β/2)
const SliceofProjection = round(Integer, Projection/Δτ)
const SliceofMeasure = round(Integer, parms.measure[1]/Δτ)
const latticetype = parms.latticetype[1]

# Log File, standard
if latticetype==3
    laname = "Pi"
elseif latticetype==2
    laname = "Sqr"
elseif latticetype==4
    laname = "4Pi"
else
    throw(DomainError("lattice type"))
end

# Log File, standard
const datafile = "Bond$(laname) N$(N)Ns$(Ns) L$(Row)X$(Col) doping$(doping) gsbd$(U[1])gtcur$(U[2]) W$(WarmStep)M$(MeasureStep) T$(β)I$(Interval)TM$(SliceofMeasure) P$(P0)"

if icpurank==rootcpurank
    # Log File, output measurements
    if !isdir(datafile)
        mkdir(datafile)
    end
    isdir(joinpath(datafile,"log")) || mkdir(joinpath(datafile,"log"))
end
MPI.Barrier(comm)

if icpurank==rootcpurank 
    ilogfile = open(joinpath(datafile,"log/log-$icpurank.log"),"w")
    redirect_stdout(ilogfile)
end
ierrfile = open(joinpath(datafile,"log/error-$icpurank.log"),"w")
redirect_stderr(ierrfile)


println("Start the simulation on cpu $icpurank")

println(parms[1])

println("Basic parameters are initialized")


println("--- Initializing the kinetic Hamiltonian ---")
# BondGroup[ib,:,ig] = (sitei,sitej) collects all the NN bonds (in 4 groups) of the lattice.
# K is the hopping term which is seperated into the ↑ part and the ↓ part. 
# eK represents the exponential of K.
if latticetype==2
    const K, BondGroup, SiteMap, Nsite, Nlayer = MakeSquare()
elseif latticetype==4
    const K, BondGroup, SiteMap, Nsite, Nlayer = Make4Site()
end
const eK = exp(-Δτ*K)
const eKHalf = exp(-Δτ*K/2.0)
const eMK = exp(Δτ*K)
const eMKHalf = exp(Δτ*K/2.0)
println(BondGroup)
println("K is initialized")


println("--- Initializing the trail wave function |psi> ---")
if latticetype==2
    const Slater, SlaterDagger = MakeSquareTrailWave()
elseif latticetype==4
    const Slater, SlaterDagger = Make4Trial()
end
println("|psi> is initialized")


println("--- Initializing the auxillary field ---")
# γ[ist] is the extra phase of the HS decomposition. ist corresponds to a 4-base Ising digits.
# eVB[:,:,gi,ist] contains the exponential of interaction Hamiltonian after the HS decomposition on each bond (group gi). eMVB is the inverse matrix.
# IsingΔ[:,:,ist1,ist2] is the transformation matrix between local HS field states in terms of the interaction terms.
const Nhs = 4 # number of HS fields
const NIsingState = (4)^Nhs # local state in base-4 digits

# σ matrix
const σ0 = Matrix{_mattype}([1+0im 0; 0 1+0im])
const σ1 = Matrix{_mattype}([0 1+0im; 1+0im 0])
const σ2 = Matrix{_mattype}([0 0-1im; 0+1im 0])
const σ3 = Matrix{_mattype}([1+0im 0; 0 -1+0im])

# random seed
Random.seed!(seed+icpurank)
const γ, eVB, eMVB, IsingΔ = InitializeBondHS() # (Δτ,U,NIsingState)
# IsingField is the intial HS field on the lattice bonds.
IsingField = InitializeIsing() # (BondGroup,Interval,NIsingState)
println("HS parameters and Ising field are initialized")

println("--- Allocate the memory for expVg and expMVg ---")
# Initialize the exp(V) term
const Imat = Matrix(I,Nsite,Nsite) # useful term for constructing evg and emvg
eVg = zeros(_mattype,Nsite,Nsite)
eMVg = zeros(_mattype,Nsite,Nsite)
println("expV (memory) is allocated")
CalculateExpVg!(eVg,1,1,zeros(Int,2N))

WeightSign = ones(Integer,1)
#WeightSign[1] = convert(Integer, real(CalculateSign()))
println("initial weight sign: ", WeightSign)


println("--- Initialize the U< U> groups ---")
# Initialize the U< and U> group
Ubra, Uket = CalculateUDV(eK,eVg) # (Slater,SlaterDagger,eK,eV,BondGroup,Nsite,Interval,P0)
println("--- the U< U> groups are initialized---")

# Initialize Green function
GreenFunc = zeros(_mattype,Nsite,Nsite)
