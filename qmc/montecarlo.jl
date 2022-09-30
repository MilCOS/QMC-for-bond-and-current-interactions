# Copyright (c) 2022, Han Xu
# All rights reserved.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree. 


include("utilities.jl")


# --- Monte Carlo procedure ---
"""
Update the B> part
"""
function UpdateIsingKet!(
    direction::Integer,
    isingfield::Array{<:Integer,3},
    greenfunc::Array{_mattype,2}, 
    weightsign::Array{<:Integer,1},
    evg::Array{_mattype,2}, emvg::Array{_mattype,2},
    uket::Array{_mattype,3}, ubra::Array{_mattype,3}
    )
    if direction == +1
        ta,tb,tc = 1,Interval,1
    elseif direction == -1
        ta,tb,tc = Interval,1,Interval
    else
        println("Wrong Direction")
    end
    # pre-allocated array
    indices = Vector{Int}(undef,2N)
    gibtmp = zeros(_mattype,2N,2N)
    dgfunc = Array{_mattype}(undef,size(greenfunc))
    for t=ta:direction:tb
        # re-calculate the Green function of the last step from scratch
        if abs(t-tc)%P0==0
            gfuncold = copy(greenfunc)
            direction==+1 && CalculateGreenFunc!(t-1,direction,indices,greenfunc,evg,uket,ubra)
            direction==-1 && CalculateGreenFunc!(t,direction,indices,greenfunc,evg,uket,ubra)
            if abs(t-tc)%50P0==0 
                icpurank==rootcpurank && println("Data: ", now(), " fast update error at $t: ", maximum(map(abs, gfuncold-greenfunc)))
            end
        end
        # record the Green function for measurements
        if DoMeasure
            direction==+1 && RecordGreenFunc!(t-1,weightsign)
            direction==-1 && RecordGreenFunc!(t,weightsign)
        end
        # update the HS field on the bond
        UpdateBond!(t,direction,isingfield,greenfunc,weightsign,evg,emvg,indices,gibtmp,dgfunc)
    end
end

"""
Go through the lattice bonds of nbgroup kinds.
0->beta, then beta->0
"""
function UpdateBond!(
    time::Integer, direction::Integer,
    isingfield::Array{<:Integer,3},
    greenfunc::Array{_mattype,2}, weightsign::Array{<:Integer,1},
    evg::Array{_mattype,2}, emvg::Array{_mattype,2},
    indices::Array{<:Integer,1}, gibtmp::Array{_mattype,2}, dgfunc::Array{_mattype,2}
    )
    #NIsingState, local state in base-4 digits
    if direction == +1
        ga,gb = 1,size(BondGroup,2)
    elseif direction == -1
        ga,gb = size(BondGroup,2),1
    else
        println("Wrong Direction")
    end

    for gi = ga:direction:gb # group type
        # Calculate eVg and eMVg from scratch
        CalculateExpVg!(evg,time,gi,indices)
        CalculateExpMVg!(emvg,time,gi,indices)

        WarpUpdateR2L!(gi, direction, greenfunc, evg, emvg) # 

        for bi=1:size(BondGroup,1)
            i, j = BondGroup[bi,gi]
            for n=1:N
                indices[n] = i+(n-1)*Nlayer # i1i2i3i4
                indices[N+n] = j+(n-1)*Nlayer # j1j2j3j4
            end
            oldist = isingfield[bi,gi,time] # ising state
            #isingbit = digits(ist-1, base=4, pad=4) .+ 1
            newist = copy(oldist)
            while true
                newist = rand(1:NIsingState)
                newist!=oldist && break
            end

            detR = CalculateMetroplis(oldist, newist, indices, greenfunc, gibtmp)
            detR = detR^Ns

            metroplis = min(1, γ[newist]/γ[oldist] * real(detR))

            # --- update ---
            if abs(metroplis) >= rand()
                if (metroplis < 0)
                    global minus_sign += 1
                end
                if metroplis < 0
                    weightsign[1] *= -1
                end
                isingfield[bi,gi,time] = newist
                UpdateExpV!(newist, indices, evg, emvg)
                UpdateLocal!(oldist, newist, indices, greenfunc, gibtmp, dgfunc)
            end
        end

        WarpUpdateL2R!(gi, direction, greenfunc, evg, emvg) #

    end
end


# --- Monte Carlo update ---
"""
Calculating the metropolis probability by
    input: oldisingstate, newisingstate, and the indices of [i↑i↓j↑j↓]
    output: probability #prob .ie. det[1+Δ(1-G)]
    output: and inverse of matrix (1+Δ(1-G)) to be #prob_inv
"""
function CalculateMetroplis(
    oldist::Integer,
    newist::Integer,
    indices::Array{<:Integer,1},
    greenfunc::Array{<:Number,2},
    gibtmp::Array{<:Number,2}
    )
    for a=1:2N
        for b=1:2N
            gibtmp[a,b] = greenfunc[indices[a],indices[b]]
        end
    end
    # mgfuncb = I - gibtmp
    Δ = IsingΔ[:,:,newist,oldist] #CalculateIsing(oldist,newist) # 
    R = I + (Δ - Δ*gibtmp)
    return det(R)
end

"""
Update the eVg and eMVg array after the update of Ising field
"""
function UpdateExpV!(
    newist::Integer,
    indices::Array{<:Integer,1},
    evg::Array{_mattype,2}, emvg::Array{_mattype,2}
    )
    # exp(Vb)
    for a=1:2N
        for b=1:2N
            evg[indices[a],indices[b]] = eVB[a,b,newist]
        end
    end
    # exp(-Vb)
    for a=1:2N
        for b=1:2N
            emvg[indices[a],indices[b]] = eMVB[a,b,newist]
        end
    end
end

"""
Update the Green function G(τ)
G = G - G*RD*(1-G)
"""
function UpdateLocal!(
    oldist::Integer,
    newist::Integer,
    indices::Array{<:Integer,1},
    greenfunc::Array{_mattype,2},
    gibtmp::Array{_mattype,2},
    dgfunc::Array{_mattype,2}
    )
    for a=1:2N
        for b=1:2N
            gibtmp[a,b] = greenfunc[indices[a],indices[b]]
        end
    end
    #mgfuncb = I - gibtmp
    Δ = IsingΔ[:,:,newist,oldist] #CalculateIsing(oldist,newist) #
    RΔ = inv(I+(Δ - Δ*gibtmp))*Δ
    
    #mgfunc = I - greenfunc
    #dgfunc = zeros(_mattype, size(greenfunc))
    for i=1:size(greenfunc,1)
        for j=1:size(greenfunc,2)
            dgfunc[i,j] = 0.0
            for a=1:2N
                for b=1:2N
                    dgfunc[i,j] += greenfunc[i,indices[a]]*RΔ[a,b]*(δ(indices[b],j)-greenfunc[indices[b],j])
                end
            end
        end
    end
    greenfunc[:,:] = greenfunc - dgfunc
end


"""
Update the Green function G(τ) to G(τ+Δτ) where τ=nΔτ.
Because in this problem the forward-backward schedule must be treated differently, 
we write the following two functions. 
When adopting the forward schedule, 
the Green function G=1-B>(B<B>)^{-1}B< must be multiplied by exp(Vb) on the left hand side
and exp(-Vb) on the right-hand side where Vb is one kind of lattice bond. 
In this way, we would be able to have the relation B>'=(Δ+1)B> when performing MC update. 
Additionally, the interaction part consists of exp(K)exp(V4)exp(V3)exp(V2)exp(V1).
"""
function WarpUpdateR2L!(
    groupid::Integer, direction::Integer,
    greenfunc::Array{_mattype,2},
    evg::Array{_mattype,2}, emvg::Array{_mattype,2}
    )
    nbond, nbgroup = size(BondGroup)

    if direction == +1
        # When scanning forwards, we must multiply 
        # exp(V1) to the left-hand side (for later MC update on V1), and exp(-V1) to the right-hand side.
        # GreenFunc = evg * GreenFunc * emvg. (B<V)B> -> B<(VB>)
        greenfunc[:,:] = BLAS.gemm('N','N',evg,greenfunc)
        greenfunc[:,:] = BLAS.gemm('N','N',greenfunc,emvg)
    elseif direction == -1  # L2R
        # When scanning backwards, 
        # Before doing the MC update on V4, we must clear the kinetic part.
        # GreenFunc = eMK * GreenFunc * eK (given that time is not 1)  B<(KB>) -> (B<K)B>
        if groupid == nbgroup
            greenfunc[:,:] = BLAS.gemm('N','N', BLAS.gemm('N','N',eMK,greenfunc), eK)
        end
    else
        println("Wrong Direction")
    end
end

function WarpUpdateL2R!(
    groupid::Integer, direction::Integer,
    greenfunc::Array{_mattype,2},
    evg::Array{_mattype,2}, emvg::Array{_mattype,2}
    )
    nbond, nbgroup = size(BondGroup)
    
    if direction == +1 # R2L
        # For the forward-update, however, after multiplying the fourth groupid, the kinetic part should be removed as well.
        # GreenFunc = eK * GreenFunc * eMK. (B<K)B> -> B<(KB>)
        if groupid == nbgroup
            greenfunc[:,:] = BLAS.gemm('N','N', BLAS.gemm('N','N',eK,greenfunc), eMK)
        end
    elseif direction == -1
        # For the backward-update, after the MC update on V4, we must multiply exp(-V4) to the left-hand side (to clear the way for MC update on V3), and exp(V4) to the right-hand side.
        # GreenFunc = emvg * GreenFunc * evg. B<(VB>) -> (B<V)B>
        greenfunc[:,:] = BLAS.gemm('N','N',emvg,greenfunc)
        greenfunc[:,:] = BLAS.gemm('N','N',greenfunc,evg)
    else
        println("Wrong Direction")
    end
end
