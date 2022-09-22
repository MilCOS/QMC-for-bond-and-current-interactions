# Copyright (c) 2022, Han Xu
# All rights reserved.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree. 


function MeasureEqualTime!(
    weightsign::Array{<:Integer,1},
    energy::Array{_mattype,1},
    mgfunc::Array{_mattype,2},
    corsbd::Array{_mattype,3},
    cortbd::Array{_mattype,3},
    corscur::Array{_mattype,3},
    cortcur::Array{_mattype,3},
    cordd::Array{_mattype,2},
    corpp::Array{_mattype,2},
    corss::Array{_mattype,2}
    )
    # pre-allocated array
    gfunc = copy(GreenFunc)
    weight_sign = copy(weightsign[1])
    # second-order Trotter. exp(K)exp(V) -> exp(K/2)exp(V)exp(K/2)
    #gfunc = BLAS.gemm('N','N', BLAS.gemm('N','N',eMKHalf,gfunc), eKHalf)
    # measure
    mgfunc[:,:] += gfunc * weight_sign
    MeasureEnergy!(gfunc,weight_sign,energy)
    MeasureCorrelationBond!(gfunc,weight_sign,corsbd,cortbd,corscur,cortcur)
    MeasureCorrelationSite!(gfunc,weight_sign,cordd,corpp,corss)
end



function δ(a::Integer,b::Integer)
    d = a==b ? 1 : 0
    d
end


# --- basic measurement functions ---
"""
Ground state energy: kinetic energy, interaction parts
"""
function MeasureEnergy!(
    gfunc::Array{_mattype,2},weight_sign::Integer,
    energy::Array{_mattype,1}
    )
    nbond, nbgroup = size(BondGroup)
    # start from bond
    for gi=1:nbgroup
        for bi=1:nbond
            i, j = BondGroup[bi,gi]            
            # kinetic energy
            for n=1:N
                a1 = i + (n-1)*Nlayer
                a2 = j + (n-1)*Nlayer
                energy[1] += - (gfunc[a2,a1]+gfunc[a1,a2]) * K[i,j] * weight_sign # -t*(-1) = 1
            end
            # singlet bond
            energy[2] += (-U[1]/2) * bondxbond(gfunc, [σ0], i,j,i,j) * weight_sign
            # triplet current
            energy[3] += (-U[2]/2) * (1/4) * currxcurr(gfunc,[σ1,σ2,σ3], i,j,i,j) * weight_sign
        end
    end
end

"""
Correlation functions defined on lattice bond
"""
function MeasureCorrelationBond!(
    gfunc::Array{_mattype,2},weight_sign::Integer,
    corsbd::Array{_mattype,3},cortbd::Array{_mattype,3},
    corscur::Array{_mattype,3},cortcur::Array{_mattype,3}
    )
    nbond, nbgroup = size(BondGroup)
    dimer = 0.0
    current = 0.0
    # start from site
    for x0=1:Col
        for y0=1:Row
            i = SiteMap[x0,y0] # bond1
            xx0 = x0==Col ? 1 : x0+1
            yy0 = y0==Row ? 1 : y0+1
            # i,i+x,i+y
            jx = SiteMap[xx0,y0]
            jy = SiteMap[x0,yy0]
            for x1=1:Col
                for y1=1:Row
                    k = SiteMap[x1,y1] # bond2
                    xx1 = x1==Col ? 1 : x1+1
                    yy1 = y1==Row ? 1 : y1+1
                    # k,k+x,k+y
                    lx = SiteMap[xx1,y1]
                    ly = SiteMap[x1,yy1]
                    
                    dimer = bondxbond(gfunc, [σ0], i,jx,k,lx) * K[i,jx]*K[k,lx] * weight_sign # xx
                    corsbd[i,k,1] += dimer
                    dimer = bondxbond(gfunc, [σ0], i,jx,k,ly) * K[i,jx]*K[k,ly] * weight_sign # xy
                    corsbd[i,k,2] += dimer
                    dimer = bondxbond(gfunc, [σ0], i,jy,k,lx) * K[i,jy]*K[k,lx] * weight_sign # yx
                    corsbd[i,k,3] += dimer
                    dimer = bondxbond(gfunc, [σ0], i,jy,k,ly) * K[i,jy]*K[k,ly] * weight_sign # yy
                    corsbd[i,k,4] += dimer

                    dimer = bondxbond(gfunc, [σ1,σ2,σ3], i,jx,k,lx) * K[i,jx]*K[k,lx] * weight_sign # xx
                    cortbd[i,k,1] += dimer
                    dimer = bondxbond(gfunc, [σ1,σ2,σ3], i,jx,k,ly) * K[i,jx]*K[k,ly] * weight_sign # xy
                    cortbd[i,k,2] += dimer
                    dimer = bondxbond(gfunc, [σ1,σ2,σ3], i,jy,k,lx) * K[i,jy]*K[k,lx] * weight_sign # yx
                    cortbd[i,k,3] += dimer
                    dimer = bondxbond(gfunc, [σ1,σ2,σ3], i,jy,k,ly) * K[i,jy]*K[k,ly] * weight_sign # yy
                    cortbd[i,k,4] += dimer

                    current = currxcurr(gfunc, [σ0], i,jx,k,lx) * K[i,jx]*K[k,lx] * weight_sign # xx
                    corscur[i,k,1] += current
                    current = currxcurr(gfunc, [σ0], i,jx,k,ly) * K[i,jx]*K[k,ly] * weight_sign # xy
                    corscur[i,k,2] += current
                    current = currxcurr(gfunc, [σ0], i,jy,k,lx) * K[i,jy]*K[k,lx] * weight_sign # yx
                    corscur[i,k,3] += current
                    current = currxcurr(gfunc, [σ0], i,jy,k,ly) * K[i,jy]*K[k,ly] * weight_sign # yy
                    corscur[i,k,4] += current

                    current = currxcurr(gfunc, [σ1,σ2,σ3], i,jx,k,lx) * K[i,jx]*K[k,lx] * weight_sign # xx
                    cortcur[i,k,1] += current
                    current = currxcurr(gfunc, [σ1,σ2,σ3], i,jx,k,ly) * K[i,jx]*K[k,ly] * weight_sign # xy
                    cortcur[i,k,2] += current
                    current = currxcurr(gfunc, [σ1,σ2,σ3], i,jy,k,lx) * K[i,jy]*K[k,lx] * weight_sign # yx
                    cortcur[i,k,3] += current
                    current = currxcurr(gfunc, [σ1,σ2,σ3], i,jy,k,ly) * K[i,jy]*K[k,ly] * weight_sign # yy
                    cortcur[i,k,4] += current
                end
            end
        end
    end
end


"""
Correlation functions defined on lattice site
"""
function MeasureCorrelationSite!(
    gfunc::Array{_mattype,2},weight_sign::Integer,
    cordd::Array{_mattype,2},
    corpp::Array{_mattype,2},
    corss::Array{_mattype,2}
    )
    for i=1:Nlayer
        for j=1:Nlayer
            # density-density correlation
            cordd[i,j] += sitexsite(gfunc,[σ0],i,j) * weight_sign
            # pair-pair correlation
            corpp[i,j] += pairxpair(gfunc,[σ1],i,j) * weight_sign
            # spin-spin correlation
            corss[i,j] += sitexsite(gfunc,[σ1,σ2,σ3],i,j) * weight_sign
        end
    end
end


"""
Calculate the bond-bond correlation. cmats represents the core matrix, C, in ψ'*C*ψ
dimer*dimer = (ij+ji)*(kl+lk)
curr*curr = (ij-ji)*(kl-lk)
"""
function bondxbond(
    gfunc::Array{_mattype,2}, cmats::Array{Array{_mattype,2},1},#Array{Symbol,1},
    i::Integer,j::Integer,k::Integer,l::Integer
    )
    dimer = 0.0
    for ci in 1:length(cmats)
        #cmat = eval(csym)
        for n1=1:N
            a1 = i+(n1-1)*Nlayer
            for n2=1:N
                a2 = j+(n2-1)*Nlayer
                for m1=1:N
                    b1 = k+(m1-1)*Nlayer
                    for m2=1:N
                        b2 = l+(m2-1)*Nlayer
                        if cmats[ci][n1,n2]*cmats[ci][m1,m2]==0
                            dimer += 0
                        else
                            coeff = cmats[ci][n1,n2]*cmats[ci][m1,m2]
                            dimer += coeff * wick(gfunc,a1,a2,b1,b2)

                            coeff = cmats[ci][n2,n1]*cmats[ci][m1,m2]
                            dimer += coeff * wick(gfunc,a2,a1,b1,b2)

                            coeff = cmats[ci][n1,n2]*cmats[ci][m2,m1]
                            dimer += coeff * wick(gfunc,a1,a2,b2,b1)

                            coeff = cmats[ci][n2,n1]*cmats[ci][m2,m1]
                            dimer += coeff * wick(gfunc,a2,a1,b2,b1)
                        end
                    end
                end
            end
        end
    end
    return dimer
end
function currxcurr(
    gfunc::Array{_mattype,2}, cmats::Array{Array{_mattype,2},1},#Array{Symbol,1},
    i::Integer,j::Integer,k::Integer,l::Integer
    )
    current = 0.0
    for ci in 1:length(cmats)
        #cmat = eval(csym)
        for n1=1:N
            a1 = i+(n1-1)*Nlayer
            for n2=1:N
                a2 = j+(n2-1)*Nlayer
                for m1=1:N
                    b1 = k+(m1-1)*Nlayer
                    for m2=1:N
                        b2 = l+(m2-1)*Nlayer
                        if cmats[ci][n1,n2]*cmats[ci][m1,m2]==0
                            current += 0
                        else
                            coeff = cmats[ci][n1,n2]*cmats[ci][m1,m2]
                            current += coeff * wick(gfunc,a1,a2,b1,b2)*-1

                            coeff = cmats[ci][n2,n1]*cmats[ci][m1,m2]
                            current += coeff * (-wick(gfunc,a2,a1,b1,b2))*-1

                            coeff = cmats[ci][n1,n2]*cmats[ci][m2,m1]
                            current += coeff * (-wick(gfunc,a1,a2,b2,b1))*-1

                            coeff = cmats[ci][n2,n1]*cmats[ci][m2,m1]
                            current += coeff * wick(gfunc,a2,a1,b2,b1)*-1
                        end
                    end
                end
            end
        end
    end
    return current
end


"""
Calculate the density-density correlation and spin-spin correlation on each site
"""
function sitexsite(gfunc::Array{_mattype,2}, cmats::Array{Array{_mattype,2},1}, i::Integer, j::Integer)
    corr = 0.0
    for ci in 1:length(cmats)
        #cmat = eval(csym)
        for n1=1:N
            a1 = i+(n1-1)*Nlayer
            for n2=1:N
                a2 = i+(n2-1)*Nlayer
                for m1=1:N
                    b1 = j+(m1-1)*Nlayer
                    for m2=1:N
                        b2 = j+(m2-1)*Nlayer

                        coeff = cmats[ci][n1,n2]*cmats[ci][m1,m2]
                        if coeff==0
                            corr += 0
                        else
                            corr += coeff * wick(gfunc,a1,a2,b1,b2)
                        end
                    end
                end
            end
        end
    end
    return corr
end


"""
Calculate the pair-pair correlation on each site ψ'Rψ
ia' ib' jc jd -> (ia'jd)(ib'jc) - (ia'jc)(ib'jd), with c=b, d=a
"""
function pairxpair(gfunc::Array{_mattype,2}, cmats::Array{Array{_mattype,2},1}, i::Integer, j::Integer)
    corr = 0.0
    for ci in 1:length(cmats)
        #cmat = eval(csym)
        for n1=1:N
            a1 = i+(n1-1)*Nlayer
            for n2=1:N
                a2 = i+(n2-1)*Nlayer
                b1 = j+(n2-1)*Nlayer
                b2 = j+(n1-1)*Nlayer
                coeff = cmats[ci][n1,n2]*cmats[ci][n2,n1]
                if coeff==0
                    corr += 0
                else
                    corr += coeff * wickpp(gfunc,a1,a2,b1,b2)
                end
            end
        end
    end
    return corr
end


"""
<c'c c'c>
"""
function wick(gfunc::Array{_mattype,2},a::Integer,b::Integer,c::Integer,d::Integer)
    return Ns^2*(δ(a,b)-gfunc[b,a])*(δ(c,d)-gfunc[d,c]) + Ns*gfunc[b,c]*(δ(a,d)-gfunc[d,a])
end
"""
<c'c' cc>
"""
function wickpp(gfunc::Array{_mattype,2},a::Integer,b::Integer,c::Integer,d::Integer)
    return Ns*(δ(a,d)-gfunc[d,a])*(δ(b,c)-gfunc[c,b]) - Ns*(δ(a,c)-gfunc[c,a])*(δ(b,d)-gfunc[d,b])
end
