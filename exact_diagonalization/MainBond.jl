# Copyright (c) 2022, Han Xu
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 


using LinearAlgebra
using CSV
using Printf

include("Operators.jl")
include("Lattices.jl")

## parameters
params = CSV.File("parameter.csv", delim=" ", comment="#", transpose=true)
mu = params.mu # chemical potential
gsbd, gscur = params.gsbd[1], params.gscur[1] # singlet interaction
gtbd, gtcur = params.gtbd[1], params.gtcur[1] # triplet interaction
Jc = gsbd + 3gtcur/4 #
V = gsbd/2 + 3gtcur/8 #
Js = 2gsbd - gtcur/2 #
#Jc = 2(gsbd+gscur) + 3(gtbd+gtcur)
#V = (gsbd-gscur)/2 - 3(gtbd-gtcur)/4
#Js = 2(gsbd-gscur) + (gtbd-gtcur)

println("parameters:\n",params)
println("Jc: ",Jc," V: ",V, " Js: ", Js)

L = 2 # lattice size
nflavor = 2 # up, down
println("--- size of the lattice and No. of flavors ---")
println(L, nflavor)

klayer, bond = Make4Site() # hopping among lattice
nlayer = size(klayer,1) # number of sites
display(bond)
println("--- No. of sites ---")
println(nlayer)
nsite = nlayer * nflavor # number of site*nflavor
kmat = zeros(nsite,nsite) # hopping among all flavors
for a=1:nflavor
    kmat[(a-1)*nlayer+1:a*nlayer, (a-1)*nlayer+1:a*nlayer] = klayer
end
println("--- kinetic term ---")
display(kmat)
println("")

allFocks, allnelecs = MakeFock(L*L, nflavor) # all possible Fock basis
println("--- size of the Fock basis ---")
println(length(allFocks))


### zero-temperature; half-filling
filling = floor(Integer, nsite/2.0)
println("filling: ", filling)
nfocks = sum(allnelecs .== filling)
# select the half-filling basis: Focks
println("--- selecting focks ---")
Focks = SelectFocks(allFocks, allnelecs, filling)
# make fock basis dic. state => idx
println("--- making fock id ---")

# Indexing the Fock states
GetFKey(fock::AbstractArray{LHbasis,1}) = prod(["$(fock[i].state)" for i=1:length(fock)])

FocksID = Dict{String, Integer}()
for n=1:nfocks
    fock = Focks[n]
    fkey = GetFKey(fock)
    FocksID[fkey] = n
end
## construct the hamiltonian
println(Focks[1])
println("--- construct the hamiltonian ---")
global hmat = zeros(Float64,(nfocks,nfocks))
println("--- size of the hamiltonian ---")
println(size(hmat))

# iterate over the Fock basis
for n=1:nfocks
    # hopping term
    for a=1:nflavor
        for bid=1:length(bond)
            i,j = bond[bid]
            # make hopping operators
            fock = deepcopy(Focks[n]) # fock state; xxxx must use deepcopy due to immutable struct
            c_j = FermionAnnih(a, j, nflavor)
            cd_j = FermionCreate(a, j, nflavor)
            c_i = FermionAnnih(a, i, nflavor)
            cd_i = FermionCreate(a, i, nflavor)
            # act on the fock state
            #println(j,"->",i)
            phase1, fock1 = (cd_i * c_j) * fock # extra sign
            #println(i,"->",j)
            phase2, fock2 = (cd_j * c_i) * fock 
            # hopping term
            if phase1 != 0
                fkey1 = GetFKey(fock1)
                m1 = FocksID[fkey1]
                hmat[m1,n] += phase1 * klayer[i,j]
            end
            # hopping term
            if phase2 != 0
                fkey2 = GetFKey(fock2)
                m2 = FocksID[fkey2]
                hmat[m2,n] += phase2 * klayer[i,j]
            end
        end
    end
end

for n=1:nfocks
    for bid=1:length(bond)
        i,j = bond[bid]
        fock = deepcopy(Focks[n]) # fock state
        c_jup = FermionAnnih(1, j, nflavor)
        c_jdn = FermionAnnih(2, j, nflavor)
        cd_jup = FermionCreate(1, j, nflavor)
        cd_jdn = FermionCreate(2, j, nflavor)
        c_iup = FermionAnnih(1, i, nflavor)
        c_idn = FermionAnnih(2, i, nflavor)
        cd_iup = FermionCreate(1, i, nflavor)
        cd_idn = FermionCreate(2, i, nflavor)
        # interaction term, Jc
        # act on the fock state
        phase1, fock1 = (cd_iup*cd_idn*c_jdn*c_jup)*fock
        if phase1 != 0 # the Pauli exclusive principle
            # find the index of the new fock basis
            fkey1 = GetFKey(fock1)
            m1 = FocksID[fkey1]
            # Jc term
            hmat[m1,n] += -Jc * phase1
        end
        phase2, fock2 = (cd_jup*cd_jdn*c_idn*c_iup)*fock
        if phase2 != 0 # the Pauli exclusive principle
            # find the index of the new fock basis
            fkey2 = GetFKey(fock2)
            m2 = FocksID[fkey2]
            # Jc term
            hmat[m2,n] += -Jc * phase2
        end

        # interaction term, V
        n_jup = FermionNumber(1, j, nflavor)
        n_jdn = FermionNumber(2, j, nflavor)
        n_iup = FermionNumber(1, i, nflavor)
        n_idn = FermionNumber(2, i, nflavor)
        # act on the fock state
        phase1 = (n_jup * fock + n_jdn * fock)
        phase2 = (n_iup * fock + n_idn * fock)
        phase = (phase1-1) * (phase2-1)
        # V term
        hmat[n,n] += V * phase

            
        # interaction term, Js
        # SxSx+SySy = (S+S- + S-S+)/2 acts on the fock state
        phase, fock1 = (cd_iup * c_idn * cd_jdn * c_jup) * fock
        if phase != 0 # the Pauli exclusive principle
            fkey = GetFKey(fock1)
            m = FocksID[fkey]
            # Js(S+S-)/2 term
            hmat[m,n] += Js * phase / 2
        end
    
        phase, fock1 = (cd_idn * c_iup * cd_jup * c_jdn) * fock # phase1*phase2*phase3*phase4
        if phase != 0 # the Pauli exclusive principle
            fkey = GetFKey(fock1)
            m = FocksID[fkey]
            # Js(S-S+)/2 term
            hmat[m,n] += Js * phase / 2
        end
    
        # SzSz=(nup-ndn)(nup-ndn)/4 acts on the fock state
        fock = Focks[n] # fock state; no need to use deepcopy
        phase1 = (n_jup * fock - n_jdn * fock)
        phase2 = (n_iup * fock - n_idn * fock)
        phase = phase1 * phase2
        # Js(SzSz)/4 term
        hmat[n,n] += Js * phase / 4.0
    end
end

println("--- is hermitian ---")
println(ishermitian(hmat))

F = eigen(hmat)

println("--- the four lowest eigenvalues ---")
println(F.values[1:4] ./ nlayer)
println("add a constant, V=$V")
println(F.values[1:4] ./ nlayer .- V)


println("--- ground state property; Green's function <c_i cd_j>---")
global GreenFunc = zeros((nsite,nsite))
ground_coeff = F.vectors[:,1]
for n=1:nfocks
    for a=1:nflavor
        for b=1:nflavor
            for i=1:nlayer
                for j=1:nlayer
                    ia = i + (a-1)*nlayer
                    jb = j + (b-1)*nlayer
                    fock = deepcopy(Focks[n]) # fock state; must use deepcopy due to immutable struct
                    c_ia = FermionAnnih(a, i, nflavor)
                    cd_jb = FermionCreate(b, j, nflavor)
                    phase1 = cd_jb * fock
                    phase2 = c_ia * fock
                    phase = phase1 * phase2
                    if phase != 0
                        fkey = ""
                        for i=1:nlayer
                            fkey = fkey * "$(fock[i].state)"
                        end
                        m = FocksID[fkey]
                        # G_ij = <ci_a cd_jb>
                        Cn = ground_coeff[n]
                        Cm = ground_coeff[m]
                        GreenFunc[ia,jb] = conj(Cm)*phase*Cn
                    end
                end
            end
        end
    end
end


#println("--- write Green function ---")
#fname = @sprintf("gfunc_gsbd%.1f_gscur%.1f_gtbd%.1f_gtcur%.1f.csv", gsbd, gscur, gtbd, gtcur)
#fout = open("data/$(fname)","w")
#for ia=1:nsite
#    for jb=1:nsite
#        gfuncij = GreenFunc[ia,jb]
#        write(fout, "$ia $jb $gfuncij\n")
#    end
#end
#close(fout)
println("--- END ---")