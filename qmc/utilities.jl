# Copyright (c) 2022, Han Xu
# All rights reserved.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree. 


# ----- Spin-1/2 -----
"""
Initialize the HS decomposition on lattice bond
    Δτ::AbstractFloat,
    u::Array{<:AbstractFloat,1},
    nisingstate::Integer
"""
function InitializeBondHS()
    # HS parameters
    a = sqrt(6)
    γi = [1-a/3, 1+a/3, 1+a/3, 1-a/3] # i stands for ising=-2,-1,+1,+2
    ηi = map(sqrt, [2*(3+a), 2*(3-a), 2*(3-a), 2*(3+a)])
    ηi[1:2] .*= -1
    ηi[3:4] .*= +1

    # --- γ phase and eVB ---
    #NIsingState local state in base-4 digits
    evb = zeros(_mattype,2N,2N,NIsingState) # 2N comes from i1,j1,i2,j2,...,iN,j
    emvb = zeros(_mattype,2N,2N,NIsingState)
    γ = zeros(AbstractFloat,NIsingState)
    for ist=1:NIsingState
        isingbit = digits(ist-1, base=4, pad=Nhs) .+ 1
        # singlet bond
        hsbd = σ0 .* (ηi[isingbit[1]]*sqrt(U[1]*Δτ/(2Ns))) # I for Diagonal([1,1])
        hsbd4 = [0I  hsbd; hsbd  0I]
        ehsbd = exp(hsbd4)
        # triplet current
        htcur = [σ1 .* ηi[isingbit[2]], σ2 .* ηi[isingbit[3]], σ3 .* ηi[isingbit[4]]] .* (sqrt(U[2]*Δτ/(2Ns))*1im/2)
        ehtcurx = exp([0I  htcur[1]; -htcur[1]  0I])
        ehtcury = exp([0I  htcur[2]; -htcur[2]  0I])
        ehtcurz = exp([0I  htcur[3]; -htcur[3]  0I])
        evbmat = ehsbd * ehtcurx * ehtcury * ehtcurz
        # store exp(vmat).
        evb[:,:,ist] = evbmat
        emvb[:,:,ist] = inv(evb[:,:,ist])
        # γ phase
        γ[ist] = prod(γi[isingbit])
    end
    # Δ = eVi(new)/eVi(old) - 1 so that B(new)=(Δ+1)*B(old); τ=nΔτ
    isingΔ = zeros(ComplexF64,2N,2N,NIsingState,NIsingState) # 4 comes from i1,j1,i2,j2,i3,j3,i4,j4
    for newist=1:NIsingState
        for oldist=1:NIsingState
            # I for Diagonal(Vector{ComplexF64}([1,1,1,1]))
            isingΔ[:,:,newist,oldist] = evb[:,:,newist] * emvb[:,:,oldist] - I
        end
    end
    return γ, evb, emvb, isingΔ
end

function CalculateIsing(oldist::Integer, newist::Integer)
    # Δ = eVi(new)/eVi(old) - 1 so that B(new)=(Δ+1)*B(old); τ=nΔτ
    # I for Diagonal(Vector{ComplexF64}([1,1,1,1]))
    isingΔ = eVB[:,:,newist] * eMVB[:,:,oldist] - I
    return isingΔ
end


"""
Initialize the Ising field on lattice bonds
    bondgroup::Array{Tuple{Integer,Integer},2}
    interval::Integer,
    nisingstate::Integer
"""
function InitializeIsing()
    # field dimension
    nbond, nbgroup = size(BondGroup)
    isingfield = rand(1:NIsingState,nbond,nbgroup,Interval)
    isingfield
end
# ----- -----


# ----- Spin-1/2 -----
"""
Calculate the interaction term exp(V) and exp(-V) after HS decomposition, 
and return eV and eMV, τ-dependent matrices with size(eV)=(nsite,nsite,nbgroup,interval),
which is memory consuming.
Therefore, we will use the function ``CalculateExpVg!" instead.
    isingfield::Array{<:Integer,3},
    evb::Array{_mattype,3},emvb::Array{_mattype,3},
    bondgroup::Array{<:Any,2},
    nlayer::Integer,nsite::Integer,interval::Integer
"""
function CalculateExpV()
    # calculate the post-HS interaction term from scratch
    nbond, nbgroup = size(BondGroup)
    #_mattype = typeof(evb[1])
    ev = zeros(_mattype,Nsite,Nsite,nbgroup,Interval)
    emv = zeros(_mattype,Nsite,Nsite,nbgroup,Interval)
    for t=1:Interval
        for gi=1:nbgroup
            evg[:,:] = Imat
            emvg[:,:] = Imat
            for bi=1:nbond
                i, j = BondGroup[bi, gi]

                indices = [[i+n*Nlayer for n=0:N-1]; [j+n*Nlayer for n=0:N-1]]
                #indices = [i,Nlayer+i,2Nlayer+i,3Nlayer+i, j,Nlayer+j,2Nlayer+j,3Nlayer+j] # i1,j1,i2,j2,i3,j3,i4,j4
                # exp(Vb)
                evbi = eVB[:,:,IsingField[bi,gi,t]]
                for a=1:2N
                    for b=1:2N
                        evg[indices[a],indices[b]] = evbi[a,b]
                    end
                end
                # exp(-Vb)
                emvbi = eMVB[:,:,IsingField[bi,gi,t]]
                for a=1:2N
                    for b=1:2N
                        emvg[indices[a],indices[b]] = emvbi[a,b]
                    end
                end
            end
            ev[:,:,gi,t] = copy(evg)
            emv[:,:,gi,t] = copy(emvg)
        end
    end
    return ev, emv
end


"""
Calculate the interaction term exp(Vg) and exp(-Vg) after the HS decomposition.
Return evg and emvg are τ-independent matrices with size(eVg)=(nsite,nsite) so as to save the memory
indices is an array for efficiency.
"""
function CalculateExpVg!(evg::Array{_mattype,2}, time::Integer, groupid::Integer, indices::Array{<:Integer,1})
    # calculate the post-HS interaction term from scratch
    evg[:,:] = Imat # start from an identity matrix
    nbond, nbgroup = size(BondGroup)
    for bi=1:nbond
        i, j = BondGroup[bi, groupid]

        #indices: [i,Nlayer+i,2Nlayer+i,3Nlayer+i, j,Nlayer+j,2Nlayer+j,3Nlayer+j]
        for n=1:N
            indices[n] = i+(n-1)*Nlayer # i1,i2,i3,i4
            indices[N+n] = j+(n-1)*Nlayer # j1,j2,j3,j4
        end
        # exp(Vb)
        #evbi = eVB[:,:,IsingField[bi,groupid,time]]
        for a=1:2N
            for b=1:2N
                evg[indices[a],indices[b]] = eVB[a,b,IsingField[bi,groupid,time]] #evbi[a,b]
            end
        end
    end
end
function CalculateExpMVg!(emvg::Array{_mattype,2}, time::Integer, groupid::Integer, indices::Array{<:Integer,1})
    # calculate the post-HS interaction term from scratch
    emvg[:,:] = Imat # start from an identity matrix
    nbond, nbgroup = size(BondGroup)
    for bi=1:nbond
        i, j = BondGroup[bi, groupid]

        #indices: [i,Nlayer+i,2Nlayer+i,3Nlayer+i, j,Nlayer+j,2Nlayer+j,3Nlayer+j]
        for n=1:N
            indices[n] = i+(n-1)*Nlayer # i1,i2,i3,i4
            indices[N+n] = j+(n-1)*Nlayer # j1,j2,j3,j4
        end
        # exp(-Vb)
        #emvbi = eMVB[:,:,IsingField[bi,groupid,time]]
        for a=1:2N
            for b=1:2N
                emvg[indices[a],indices[b]] = eMVB[a,b,IsingField[bi,groupid,time]] #emvbi[a,b]
            end
        end
    end
end

# --- Stabilization, Green function ---
"""
The LQ and QR decompositions for stabilizing the matrix multiplication
B<=P^{d}B(β,τ). 
B>=B(τ,0)P. eKeV=eKeV4eV3eV2eV1.
    Q: orthogonal matrix; L: lower triangular matrix; R: upper triangular matrix.
    slater::Array{_mattype,2},
    slaterdagger::Array{_mattype,2},
    ek::Array{_mattype,2},
    evg::Array{_mattype,2},
    bondgroup::Array{<:Any,2},
    nsite::Integer,interval::Integer,p0::Integer
"""
function CalculateUDV(ek::Array{_mattype,2}, evg::Array{_mattype,2})
    nbond, nbgroup = size(BondGroup)
    bsbra = copy(SlaterDagger)
    bsket = copy(Slater)
    nugroup = round(Integer, Interval/P0)
    indices = Vector{Int}(undef,2N)
    # U>
    nelec = size(Slater,2)
    uket = zeros(_mattype,Nsite,nelec,nugroup)
    for t=1:Interval
        # eV4*eV3*eV2*eV1*B>
        for gi=1:nbgroup
            CalculateExpVg!(evg, t, gi, indices)
            bsket = BLAS.gemm('N','N', evg, bsket)
        end
        # eK*eV4*eV3*eV2*eV1*B>
        bsket = BLAS.gemm('N','N', ek, bsket)
        if t%P0==0
            # QR decomposition -> Q*D*D^{-1}R
            q,r = qr(bsket)
            #d = Diagonal(r) # diagonal matrix
            #dinv = Diagonal(1 ./ diag(r)) # 1/diagonal matrix
            #r = dinv*r
            bsket = Array(q) # noramilzed orthogonal matrix
            ugi = round(Integer, t/P0)
            uket[:,:,ugi] = copy(bsket) # <<<
            #bsket = bsket*d # Diagonal matrix * matrix
        end
    end
    # U<
    ubra = zeros(_mattype,nelec,Nsite,nugroup)
    for t=Interval:-1:1
        # B<*eK
        bsbra = BLAS.gemm('N','N', bsbra, ek)
        # B<*eK*eV4*eV3*eV2*eV1
        for gi=nbgroup:-1:1
            CalculateExpVg!(evg, t, gi, indices)
            bsbra = BLAS.gemm('N','N', bsbra, evg)
        end
        if (Interval-t+1)%P0==0
            # LQ decomposition -> LD^{-1}*D*Q
            l,q = lq(bsbra)
            #d = Diagonal(l) # diagonal matrix
            #dinv = Diagonal(1 ./ diag(l)) # 1/diagonal matrix
            #l = l*dinv # unit lower triangular matrix
            bsbra = Array(q) # noramilzed orthogonal matrix
            ugi = round(Integer, (Interval-t+1)/P0)
            ubra[:,:,ugi] = copy(bsbra) # <<<
            #bsbra = d*bsbra # Diagonal matrix * matrix
        end
    end
    return ubra, uket
end

"""
    Calculate the Green function from scratch using a stabilization method.
    The stabilized group will be updated.
    For updating forwards, only the uket group changes.
    For updating backwards, only the ubra group changes.
    We should count in the new isingfield i.e. new exp(V) within p0 time slice.
"""
function CalculateGreenFunc!(
    time::Integer,direction::Integer,indices::Array{<:Integer,1},
    greenfunc::Array{_mattype,2},
    evg::Array{_mattype,2},
    uket::Array{_mattype,3}, ubra::Array{_mattype,3}
    )
    nbond, nbgroup = size(BondGroup)
    if direction == +1
        # ubra. LQ decomposition
        if time+1 == 1
            # must be ``end-1'' because we are using the last group that has been updated, but Ubra[:,:,end] hasn't been updated along the other direction.
            bsbra = copy(ubra[:,:,end-1])
            for t=P0:-1:1
                # B<*eK
                bsbra = BLAS.gemm('N','N', bsbra, eK)
                # B<*eK*eV4*eV3*eV2*eV1
                for gi=nbgroup:-1:1
                    CalculateExpVg!(evg, t, gi, indices)
                    bsbra = BLAS.gemm('N','N', bsbra, evg)
                end
            end
            # LQ decomposition -> LD^{-1}*D*Q
            l,q = lq(bsbra)
            bsbra = Array(q) # noramilzed orthogonal matrix
            ubra[:,:,end] = copy(bsbra) # <<<
        else
            ugi = round(Integer, (Interval-time)/P0)
            #println("$direction, bra ugi $ugi and ", (Interval-time)/P0)
            bsbra = time==Interval ? copy(SlaterDagger) : copy(ubra[:,:,ugi])
        end
        # uket. QR decomposition
        if time+1 == 1 # or
            bsket = copy(Slater)
        else
            ugi = round(Integer, time/P0)
            #println("$direction, ket ugi $ugi and ", time/P0)
            bsket = ugi == 1 ? copy(Slater) : copy(uket[:,:,ugi-1])
            for t=time-P0+1:time
                # eV4*eV3*eV2*eV1*B>
                for gi=1:nbgroup
                    CalculateExpVg!(evg, t, gi, indices)
                    bsket = BLAS.gemm('N','N', evg, bsket)
                end
                # eK*eV4*eV3*eV2*eV1*B>
                bsket = BLAS.gemm('N','N', eK, bsket)
            end
            # QR decomposition -> Q*D*D^{-1}*R
            q, r = qr(bsket)
            bsket = Array(q) # noramilzed orthogonal matrix
            uket[:,:,ugi] = copy(bsket) # <<<
        end

    elseif direction == -1
        # uket. QR decomposition
        if time == Interval
            # must be ``end-1'' because we are using the last group that has been updated, but Uket[:,:,end] hasn't been updated along the other direction.
            bsket = copy(uket[:,:,end-1])
            for t=Interval-P0+1:Interval
                # eV4*eV3*eV2*eV1*B>
                for gi=1:nbgroup
                    CalculateExpVg!(evg, t, gi, indices)
                    bsket = BLAS.gemm('N','N', evg, bsket)
                end
                # eK*eV4*eV3*eV2*eV1*B>
                bsket = BLAS.gemm('N','N', eK, bsket)
            end
            # QR decomposition -> Q*D*D^{-1}*R
            q, r = qr(bsket)
            bsket = Array(q) # noramilzed orthogonal matrix
            uket[:,:,end] = copy(bsket) # <<<
        else
            ugi = round(Integer, time/P0)
            bsket = time==1 ? copy(Slater) : copy(uket[:,:,ugi])
        end
        # ubra. LQ decomposition
        if time == Interval
            bsbra = SlaterDagger
        else
            ugi = round(Integer, (Interval-time)/P0)
            bsbra = ugi == 1 ? copy(SlaterDagger) : copy(ubra[:,:,ugi-1])
            for t=time+P0:-1:time+1
                # B<*eK
                bsbra = BLAS.gemm('N','N', bsbra, eK)
                # B<*eK*eV4*eV3*eV2*eV1
                for gi=nbgroup:-1:1
                    CalculateExpVg!(evg, t, gi, indices)
                    bsbra = BLAS.gemm('N','N', bsbra, evg)
                end
            end
            # LQ decomposition -> LD^{-1}*D*Q
            l,q = lq(bsbra)
            bsbra = Array(q) # noramilzed orthogonal matrix
            ubra[:,:,ugi] = copy(bsbra) # <<<
        end
    
    else
        println("Wrong Direction, $direction")
    end
    # Green function G=1-B>(B<B>)^{-1}B<
    greenfunc[:,:] = I - bsket*inv(bsbra*bsket)*bsbra
end
# --- ---


"""
initialize weight
"""
function CalculateSign()
    nbond, nbgroup = size(BondGroup)
    evg = zeros(_mattype,Nsite,Nsite) # initialize evg
    s0 = 1
    indices = indices = Vector{Int}(undef,2N)
    for t=1:Interval
        for gi=1:nbgroup
            CalculateExpVg!(evg,t,gi,indices)
            s0 *= sign(det(evg))
        end
    end
    s0
end

# --- Save data for measurement ---
"""
Record the Green function within a small range around the projection length. 
For example, |.|.|.||.|.|.| has m=3, so we do measurements within [P-m,P+m].
"""
function RecordGreenFunc!(
    time::Integer,
    weightsign::Array{<:Integer,1}
    )
    ta = SliceofProjection - SliceofMeasure
    tb = SliceofProjection + SliceofMeasure
    if ta <= time <= tb
        MeasureEqualTime!(weightsign,Energy,MeanGreenFunc,CorFuncSbd,CorFuncTbd,CorFuncScur,CorFuncTcur,CorFuncDD,CorFuncPP,CorFuncSS)
    end
end
