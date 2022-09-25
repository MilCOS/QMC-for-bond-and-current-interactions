# Copyright (c) 2022, Han Xu
# All rights reserved.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree. 


import Base.:*

"""Fermion Operators"""
abstract type FermionOperator end

struct FermionCreate <: FermionOperator
    flavor :: Integer # [1 up, 2 down]
    siteid :: Integer
    nflavor :: Integer
end

struct FermionAnnih <: FermionOperator
    flavor :: Integer
    siteid :: Integer
    nflavor :: Integer
end

struct FermionNumber <: FermionOperator
    flavor :: Integer
    siteid :: Integer
    nflavor :: Integer
end


"""Local Hilbert Space"""
mutable struct LHbasis
    siteid :: Integer
    state :: Integer # [0,1,2,3] for [0, up, down, updown]
    nelec :: Integer
end

"""add LHstate to Fock state. length(fstate) is nsite, typeof(fstate[1]) is LHbasis"""
function AddLH!(fstate :: AbstractArray{LHbasis}, lhstate :: LHbasis)
    fstate[lhstate.siteid] = lhstate
end

"""Convertion between state numbers and bits lists"""
function bit2num(bits :: AbstractArray{<:Signed,1}; base = 2, pad = 2)
    sum(bits[i] * base^(i-1) for i = 1:pad)
end

"""Fermion creation operator acts on the Fock basis"""
function *(fo :: FermionCreate, fs :: AbstractArray{LHbasis,1})
    bit = digits(fs[fo.siteid].state, base = 2, pad = fo.nflavor)
    i0 = fo.flavor
    newbit = bit + [ i == i0 ? 1 : 0 for i=1:fo.nflavor]
    if any(newbit .> 1)
        return 0
    else
        phase = (i0 == 1) ? 1 : (-1)^(sum(bit[1:i0-1]))
        fs[fo.siteid].state = bit2num(newbit, base = 2, pad = fo.nflavor)
        fs[fo.siteid].nelec += 1
        for n=1:fo.siteid-1 # fermion operators before this site
            phase *= (-1)^(fs[n].nelec)
        end
        return phase
    end
end

"""Fermion annilation operator acts on the Fock basis"""
function *(fo :: FermionAnnih, fs :: AbstractArray{LHbasis,1})
    bit = digits(fs[fo.siteid].state, base = 2, pad = fo.nflavor)
    i0 = fo.flavor
    newbit = bit - [ i == i0 ? 1 : 0 for i=1:fo.nflavor]
    if any(newbit .< 0)
        return 0
    else
        phase = (i0 == 1) ? 1 : (-1)^(sum(bit[1:i0-1]))
        fs[fo.siteid].state = bit2num(newbit, base = 2, pad = fo.nflavor)
        fs[fo.siteid].nelec -= 1
        for n=1:fo.siteid-1 # fermion operators before this site
            phase *= (-1)^(fs[n].nelec)
        end
        return phase
    end
end

"""Fermion number operator acts on the Fock basis"""
function *(fo :: FermionNumber, fs :: AbstractArray{LHbasis,1})
    bit = digits(fs[fo.siteid].state, base = 2, pad = fo.nflavor)
    i0 = fo.flavor # number operator does not change the bit
    num = bit[i0] # the occupation number of this flavor; phase = 0
    num
end


"""Operator"""
function *(
    fos1 :: Union{FermionOperator,Array{<:FermionOperator,2}}, 
    fos2 :: Union{FermionOperator,Array{<:FermionOperator,2}}
    )
    [fos1 fos2]
end

#function *(fos1 :: FermionOperator, fos2 :: FermionOperator)
#    Operators([fos1 fos2])
#end

function *(fos :: Array{<:FermionOperator,2}, fs :: AbstractArray{LHbasis,1})
    fs0 = deepcopy(fs)
    phase = 1.0
    for i=1:length(fos)
        phasei = fos[end-i+1] * fs0
        phase *= phasei
    end
    return phase, fs0
end
