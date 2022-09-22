# Copyright (c) 2022, Han Xu
# All rights reserved.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree. 


using LinearAlgebra
using CSV
using Printf

include("Operators.jl")
include("Lattices.jl")

## parameters
#params = CSV.File("parameter.csv", delim=" ", comment="#", transpose=true)
mu = 0.0 #params.mu # chemical potential
gsbd, gscur = 0.0, 0.0 #params.gsbd[1], params.gscur[1] # singlet interaction
gtbd, gtcur = 0.0, 5.0 #params.gtbd[1], params.gtcur[1] # triplet interaction
println("parameters:\n")
println("gsbd: ", gsbd, "; gtcur: ", gtcur)

L = 2 # lattice size
nflavor = 2 # up, down
println("--- size of the lattice and No. of flavors ---")
println(L, nflavor)

klayer, bond = Make4Site() # MakeSquare(L) # hopping among lattice
nlayer = size(klayer,1) # number of sites
display(bond)
println("\n--- No. of sites ---")
println(nlayer)
nsite = nlayer * nflavor # number of site*nflavor
kmat = zeros(nsite,nsite) # hopping among all flavors
for a=1:nflavor
    kmat[(a-1)*nlayer+1:a*nlayer, (a-1)*nlayer+1:a*nlayer] = klayer
end
#println("--- kinetic term ---")
#display(kmat)
#println("")

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
            fock = deepcopy(Focks[n]) # fock state
            c_i = FermionAnnih(a, i, nflavor)
            cd_i = FermionCreate(a, i, nflavor)
            c_j = FermionAnnih(a, j, nflavor)
            cd_j = FermionCreate(a, j, nflavor)
            # act on the fock state
            #println(j,"->",i)
            chop_ij = cd_i * c_j
            chop_ji = cd_j * c_i
            phase1, fock1 = chop_ij * fock
            phase2, fock2 = chop_ji * fock
            # find the index of the new fock basis
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

    # interaction term
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
        # Singlet bond term and triplet current z
        M_ij = (cd_iup*c_jup, cd_idn*c_jdn, cd_jup*c_iup, cd_jdn*c_idn)
        # terms of sigmaz
        signz = [+1,-1,-1,+1]
        for a=1:4
            for b=1:4
                c_diag = M_ij[a] * M_ij[b]
                phase1, fock1 = c_diag * fock
                # find the index of the new fock basis
                if phase1 != 0
                    fkey1 = GetFKey(fock1)
                    m1 = FocksID[fkey1]
                    hmat[m1,n] += (-gsbd/2) * phase1
                    hmat[m1,n] += (-gtcur/2)*(-1)*phase1*(signz[a]*signz[b]/2/2)
                end
            end
        end
        # Triplet current term
        Nxy_ij = (cd_iup*c_jdn, cd_idn*c_jup, cd_jup*c_idn, cd_jdn*c_iup)
        # terms of sigmax
        signx = [+1,+1,-1,-1]
        # terms of sigmay
        signy = [-1im,+1im,+1im,-1im]
        for a=1:4
            for b=1:4
                c_tcurxy = Nxy_ij[a] * Nxy_ij[b]
                phase1, fock1 = c_tcurxy * fock
                # find the index of the new fock basis
                if phase1 != 0
                    fkey1 = GetFKey(fock1)
                    m1 = FocksID[fkey1]
                    hmat[m1,n] += (-gtcur/2)*(-1)*phase1*(signx[a]*signx[b]/2/2)
                    hmat[m1,n] += (-gtcur/2)*(-1)*phase1*(signy[a]*signy[b]/2/2)
                end
            end
        end
    end
end

println("--- is hermitian ---")
println(ishermitian(hmat))

F = eigen(hmat)

println("--- the four lowest eigenvalues ---")
println(F.values[1:4] ./ nlayer)

println("--- END ---")
