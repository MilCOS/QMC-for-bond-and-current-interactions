# Copyright (c) 2022, Han Xu
# All rights reserved.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree. 


function MakeSquare(L;t=1)
    nlayer = L*L
    hopping = zeros(nlayer,nlayer)
    bond = Array{Tuple{Integer,Integer},1}()
    for x=1:L
        for y=1:L
            i = (x-1)*L + y
            # hopping to x+1
            xx = x==L ? 1 : x+1
            yy = y
            j = (xx-1)*L + yy
            hopping[j,i] = -t
            hopping[i,j] = -t
            bond = vcat(bond, (i,j))
            # hopping to y+1
            xx = x
            yy = y==L ? 1 : y+1
            j = (xx-1)*L + yy
            hopping[j,i] = -t
            hopping[i,j] = -t
            bond = vcat(bond, (i,j))
        end
    end
    return hopping, bond
end



function Make4Site(;tnn=1)
    nlayer = 4
    # _mattype = ComplexF64
    hopping = zeros(nlayer,nlayer)
    hopping[1,3] = -tnn
    hopping[3,1] = -tnn
    hopping[1,2] = -tnn
    hopping[2,1] = -tnn
    hopping[2,4] = -tnn
    hopping[4,2] = -tnn
    hopping[3,4] = -tnn
    hopping[4,3] = -tnn
    bond = [(1,3),(1,2),(2,4),(3,4)]
    return hopping, bond
end


function Make4Pi(;tnn=1)
    nlayer = 4
    # _mattype = ComplexF64
    hopping = zeros(nlayer,nlayer)
    hopping[1,3] = -tnn
    hopping[3,1] = -tnn
    hopping[1,2] = +tnn
    hopping[2,1] = +tnn
    hopping[2,4] = -tnn
    hopping[4,2] = -tnn
    hopping[3,4] = -tnn
    hopping[4,3] = -tnn
    bond = [(1,3),(1,2),(2,4),(3,4)]
    return hopping, bond
end


"""generate the fock basis. Block-diagonal matrix depending on filling number"""
function MakeFock(nlayer,nflavor)
    nsite = nlayer*nflavor
    allbits = hcat([digits(n, base = 2, pad = nsite) for n=0:2^nsite-1]...)'
    allnelecs = sum(allbits, dims=2)
    allFocks = Array{Array{LHbasis,1}}(undef, length(allnelecs))
    for n=1:length(allnelecs)
        bits = allbits[n,:]
        st0 = Array{LHbasis}(undef,nlayer)
        for i=1:nlayer
            state = bit2num(bits[(i-1)*nflavor+1:i*nflavor], base = 2, pad = nflavor)
            nelec = sum(bits[(i-1)*nflavor+1:i*nflavor])
            AddLH!(st0, LHbasis(i, state, nelec))
        end
        allFocks[n] = st0
    end

    return allFocks, allnelecs[:,1]
end

"""select the fock basis by the filling number"""
function SelectFocks(allFocks, allnelecs, filling)
    nfocks = sum(allnelecs .== filling)
    Focks = Array{Array{LHbasis,1}}(undef, nfocks)
    i = 1
    for n=1:length(allFocks)
        if allnelecs[n] != filling # not half-filling; skip
            continue
        end
        Focks[i] = allFocks[n]
        i += 1
    end
    Focks
end




#include("operators.jl")
#allFocks, allnelecs=MakeFock(2, 3)
#println(SelectFocks(allFocks, allnelecs, 3))
