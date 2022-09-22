# Copyright (c) 2022, Han Xu
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 


using MPI
using LinearAlgebra
using Random
using Statistics
using Printf
using CSV
using HDF5
using Dates


MPI.Init()

include("initialize.jl")
include("montecarlo.jl")
include("measure.jl")
include("postprocess.jl")


# --- initialize measurement ---
# Array has mutable contents. Always use A[:] = data in the functions
# energy vs ED
Energy = zeros(_mattype, 3) #kinetic, sbd, tcur*3
MeanGreenFunc = zeros(_mattype, size(GreenFunc))
# correlation functions on bond
CorFuncSbd = zeros(_mattype, Nlayer, Nlayer, 4)
CorFuncTbd = zeros(_mattype, Nlayer, Nlayer, 4)
CorFuncScur = zeros(_mattype, Nlayer, Nlayer, 4)
CorFuncTcur = zeros(_mattype, Nlayer, Nlayer, 4)
# correlation functions on site
CorFuncDD = zeros(_mattype, Nlayer, Nlayer)
CorFuncPP = zeros(_mattype, Nlayer, Nlayer)
CorFuncSS = zeros(_mattype, Nlayer, Nlayer)
# monte carlo path
EnergyPath = zeros(_mattype, MeasureStep)
global minus_sign = 0


println("--- start QMC steps for warming up ---")
# --- warming up steps ---
DoMeasure = false
for step=1:WarmStep
    icpurank==rootcpurank && println("=== Warming up step $step ===")
    if step%2==1
        direction = +1
    else
        direction = -1
    end
    UpdateIsingKet!(direction,IsingField,GreenFunc,WeightSign,eVg,eMVg,Uket,Ubra)
end

println("--- start QMC steps for measurements ---")
# --- measurement steps ---
DoMeasure = true
for step=1:MeasureStep
    icpurank==rootcpurank && println("=== Measurement step $step ===")
    if step%2==1
        direction = +1
    else
        direction = -1
    end
    UpdateIsingKet!(direction,IsingField,GreenFunc,WeightSign,eVg,eMVg,Uket,Ubra)
    # data as a function of the monte carlo step
    EnergyPath[step] = sum(Energy) / (step)
end

println("--- enter the post process ---")


# --- average and output ---

OutputAverage(Energy,EnergyPath,MeanGreenFunc,CorFuncSbd,CorFuncTbd,CorFuncScur,CorFuncTcur,CorFuncDD,CorFuncPP,CorFuncSS)

println("--- finalize ---")

MPI.Finalize()
