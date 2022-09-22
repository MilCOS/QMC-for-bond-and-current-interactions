# Copyright (c) 2022, Han Xu
# All rights reserved.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree. 


"""
    Average and Error over the MPI processes/ Return the mean values and the variance
"""
function MPIAverage(vals::Array{<:Number,N}) where N
    vsize = size(vals)
    MPI.Barrier(comm)
    newvals = MPI.Gather(vals, rootcpurank, comm)
    if icpurank==rootcpurank
        newvals = transpose( reshape(newvals,(prod(vsize),ncpurank)) )

        mvals = mean(newvals,dims=1)
        vvals = std(newvals,mean=mvals,dims=1)

        mvals = reshape(mvals, vsize)
        vvals = reshape(vvals, vsize)

        return mvals, vvals
    else
        nothing, nothing
    end
end

"""
    Lattice Fourie transformation. Calculate the structure factors from correlation functions on the square lattice.
"""
function LatticeFourie(corfunc::Array{<:Number,2}, kx::Real, ky::Real)
    strucfactor = 0.0+0.0im
    for x=1:Row
        for y=1:Col
            for xx=1:Row
                for yy=1:Col
                    i=SiteMap[x,y]
                    j=SiteMap[xx,yy]
                    strucfactor += corfunc[i,j] * exp(1im*pi*((xx-x)*kx+(yy-y)*ky))
                end
            end
        end
    end
    strucfactor / Nlayer
end


"""
    Output data on rootcpurank
"""
function OutputAverage(
    energy::Array{_mattype,1},
    energypath::Array{_mattype,1},
    mgfunc::Array{_mattype,2},
    corsbd::Array{_mattype,3},
    cortbd::Array{_mattype,3},
    corscur::Array{_mattype,3},
    cortcur::Array{_mattype,3},
    cordd::Array{_mattype,2},
    corpp::Array{_mattype,2},
    corss::Array{_mattype,2}
    )
    nbond, nbgroup = size(BondGroup)
    # average over the Monte simulation and lattice bonds
    oenergy = energy ./ (MeasureStep * (2*SliceofMeasure+1) * (nbond*nbgroup)) ./ Ns
    oenergypath = energypath ./ ((2*SliceofMeasure+1) * (nbond*nbgroup)) ./ Ns
    ogfunc = mgfunc ./ (MeasureStep * (2*SliceofMeasure+1))
    odensity1 = [sum(Diagonal(I-ogfunc))] ./ (Nsite)
    ominussign = minus_sign / ((WarmStep+MeasureStep)*nbond*nbgroup*Interval)

    ocorsbd = corsbd ./  (MeasureStep * (2*SliceofMeasure+1)) ./ Ns
    ocortbd = cortbd ./  (MeasureStep * (2*SliceofMeasure+1)) ./ Ns
    ocorscur = corscur ./  (MeasureStep * (2*SliceofMeasure+1)) ./ Ns
    ocortcur = cortcur ./  (MeasureStep * (2*SliceofMeasure+1)) ./ Ns
    ocordd = cordd ./  (MeasureStep * (2*SliceofMeasure+1))
    ocorpp = corpp ./  (MeasureStep * (2*SliceofMeasure+1))
    ocorss = corss ./  (MeasureStep * (2*SliceofMeasure+1))

    ## order parameters
    sbdx0 = LatticeFourie(ocorsbd[:,:,1], 1,0) # singlet DW xx. (pi,0)
    sbdy0 = LatticeFourie(ocorsbd[:,:,4], 0,1) # singlet DW yy. (0.pi)
    sbd0 = sqrt((sbdx0+sbdy0)/Nlayer)
    sbdx0dx = LatticeFourie(ocorsbd[:,:,1], 1+2/Col,0)
    sbdx0dy = LatticeFourie(ocorsbd[:,:,1], 1,0+2/Row)
    sbdy0dx = LatticeFourie(ocorsbd[:,:,4], 0+2/Col,1)
    sbdy0dy = LatticeFourie(ocorsbd[:,:,4], 0,1+2/Row)
    Rsbdx_dx = 1.0 - sbdx0dx/sbdx0
    Rsbdx_dy = 1.0 - sbdx0dy/sbdx0
    Rsbdy_dx = 1.0 - sbdy0dx/sbdy0
    Rsbdy_dy = 1.0 - sbdy0dy/sbdy0
    Rsbd = 0.25*(Rsbdx_dx+Rsbdx_dy+Rsbdy_dx+Rsbdy_dy)

    tbdx0 = LatticeFourie(ocortbd[:,:,1], 1,0) # triplet DW xx. (pi,0)
    tbdy0 = LatticeFourie(ocortbd[:,:,4], 0,1) # triplet DW yy. (0.pi)
    tbd0 = sqrt((tbdx0+tbdy0)/Nlayer)

    scurx0 = LatticeFourie(ocorscur[:,:,1], 1,1) # singlet flux xx. (pi,pi)
    scurxy = LatticeFourie(ocorscur[:,:,2], 1,1) # singlet flux xy. (pi,pi)
    scuryx = LatticeFourie(ocorscur[:,:,3], 1,1) # singlet flux yx. (pi,pi)
    scury0 = LatticeFourie(ocorscur[:,:,4], 1,1) # singlet flux yy. (pi.pi)
    scur0 = sqrt((scurx0-scurxy-scuryx+scury0)/Nlayer)

    tcurx0 = LatticeFourie(ocortcur[:,:,1], 1,1) # triplet flux xx. (pi,pi)
    tcurxy = LatticeFourie(ocortcur[:,:,2], 1,1) # triplet flux xy. (pi,pi)
    tcuryx = LatticeFourie(ocortcur[:,:,3], 1,1) # triplet flux yx. (pi,pi)
    tcury0 = LatticeFourie(ocortcur[:,:,4], 1,1) # triplet flux yy. (pi.pi)
    tcur0 = sqrt((tcurx0-tcurxy-tcuryx+tcury0)/Nlayer)
    tcurtmp = ocortcur[:,:,1]-ocortcur[:,:,2]-ocortcur[:,:,3]+ocortcur[:,:,4]
    tcurdk = 0.25*(LatticeFourie(tcurtmp, 1+2/Col,1) + LatticeFourie(tcurtmp, 1-2/Col,1) + LatticeFourie(tcurtmp, 1,1+2/Row) + LatticeFourie(tcurtmp, 1,1-2/Row))
    Rtcur = 1.0 - tcurdk/(tcurx0-tcurxy-tcuryx+tcury0)

    cdw = LatticeFourie(ocordd, 1,1) # CDW. (pi,pi)
    cdwdk = 0.25*(LatticeFourie(ocordd, 1+2/Col,1) + LatticeFourie(ocordd, 1-2/Col,1) + LatticeFourie(ocordd, 1,1+2/Row) + LatticeFourie(ocordd, 1,1-2/Row))
    #cdwdk = LatticeFourie(ocordd, 1+2/Col,1+2/Row)
    Rcdw = 1.0 - cdwdk/cdw
    cdw = sqrt(cdw/Nlayer)

    pairing = LatticeFourie(ocorpp, 0,0) # Paring. (0,0)
    pairing = sqrt(pairing/Nlayer)
    fm = LatticeFourie(ocorss, 0,0) # FM
    fm = sqrt(fm/Nlayer)
    afm = LatticeFourie(ocorss, 1,1) # AFM
    afm = sqrt(afm/Nlayer)


    # Energy
    energym, energyerr = MPIAverage(oenergy)
    # Density
    density1m, density1err = MPIAverage(odensity1)
    # green function
    ogfuncm, ogfuncerr = MPIAverage(ogfunc)
    # order parameters (bond)
    obdws, obdwerrs = MPIAverage([sbd0, tcur0])
    otbdws, otbdwerrs = MPIAverage([tbd0, scur0])
    # order parameters (site)
    ocdws, ocdwerrs = MPIAverage([cdw,pairing,fm,afm])
    # Binder ratio
    oRatio, oRatioerr = MPIAverage([Rsbd,Rtcur,Rcdw])

    # bond correlation function
    ocorsbdm, ocorsbderr = MPIAverage(ocorsbd)
    ocortbdm, ocortbderr = MPIAverage(ocortbd)
    ocorscurm, ocorscurerr = MPIAverage(ocorscur)
    ocortcurm, ocortcurerr = MPIAverage(ocortcur)
    # site correlation function
    ocorddm, ocordderr = MPIAverage(ocordd)
    ocorppm, ocorpperr = MPIAverage(ocorpp)
    ocorssm, ocorsserr = MPIAverage(ocorss)

    # data E vs QMC steps
    oepath, oepatherr = MPIAverage(oenergypath)
    # minus sign
    osignm, osignerr = MPIAverage([ominussign])

    if icpurank==rootcpurank
        # Log File, output measurements
        if !isdir(datafile)
            mkdir(datafile)
        end
        Ωfile = open( joinpath(datafile,"result.txt"),"w")
        # System
        println(Ωfile, "# Lattice")
        println(Ωfile, "row, $Row")
        println(Ωfile, "col, $Col")
        println(Ωfile, "N, $N")
        println(Ωfile, "Ns, $Ns")
        println(Ωfile, "U, "*"$U"[2:end-1])
        println(Ωfile, "mu, $μ")
        println(Ωfile, "doping, $doping")
        println(Ωfile, "# Monte Carlo")
        println(Ωfile, "warmstep, $WarmStep")
        println(Ωfile, "measurestep, $MeasureStep")
        println(Ωfile, "randomseed, $seed")
        println(Ωfile, "beta, $β")
        println(Ωfile, "interval, $Interval")
        println(Ωfile, "p0, $P0")
        println(Ωfile, "tau, $Δτ")
        println(Ωfile, "sliceofprojection, $SliceofProjection")
        println(Ωfile, "sliceofmeasure, $SliceofMeasure")
        println(Ωfile, "ncpu, $ncpurank")

        println(Ωfile, "# Density 1")
        println(Ωfile, "$(real(density1m))"[2:end-1])
        println(Ωfile, "# Density 1 Error")
        println(Ωfile, "$(real(density1err))"[2:end-1])
        println(Ωfile, "# Energy: kinetic, singlet bond, triplet current")
        println(Ωfile, "$(real(energym))"[2:end-1])
        println(Ωfile, "# Energy Error")
        println(Ωfile, "$(real(energyerr))"[2:end-1])
        println(Ωfile, "# Bond order parameters: singlet vbs bond, triplet flux")
        println(Ωfile, "$(real(obdws))"[2:end-1])
        println(Ωfile, "# Bond order parameters Error")
        println(Ωfile, "$(real(obdwerrs))"[2:end-1])
        println(Ωfile, "# Order: cdw, pairing, fm, afm")
        println(Ωfile, "$(real(ocdws))"[2:end-1])
        println(Ωfile, "# Order Error cdw, pairing, fm, afm")
        println(Ωfile, "$(real(ocdwerrs))"[2:end-1])
        println(Ωfile, "Average minus sign")
        println(Ωfile, "$(osignm)"[2:end-1])
        println(Ωfile, "Average minus sign Error")
        println(Ωfile, "$(osignerr)"[2:end-1])
        println(Ωfile, "# Bond order parameters: triplet vbs bond, singlet flux")
        println(Ωfile, "$(real(otbdws))"[2:end-1])
        println(Ωfile, "# Bond order parameters Error (Triplet)")
        println(Ωfile, "$(real(otbdwerrs))"[2:end-1])
        println(Ωfile, "# Binder Ratio: singlet bond-1, triplet current-1, cdw-1")
        println(Ωfile, "$(real(oRatio))"[2:end-1])
        println(Ωfile, "# Binder Ratio Error")
        println(Ωfile, "$(real(oRatioerr))"[2:end-1])
        flush(Ωfile)
        close(Ωfile)

        # Correlation File. Greenfunc, bond, site
        #Gfile = open( joinpath(datafile,"gfunc.txt"),"w")
        #for i=1:Nsite
        #    for j=1:Nsite
        #        @printf(Gfile, "%i, %i, %16.f10, %16.f10, %16.f10, %16.f10\n", i, j,
        #            real(ogfuncm[i,j]),imag(ogfuncm[i,j]),real(ogfuncerr[i,j]),imag(ogfuncerr[i,j])
        #        )
        #    end
        #end
        corfid = h5open( joinpath(datafile,"correlation.h5"), "w")
        create_group(corfid, "data")
        #create_dataset(）
        corfid["data"]["greenfunction"] = ogfuncm
        corfid["data"]["singletbond"] = ocorsbdm
        corfid["data"]["tripletbond"] = ocortbdm
        corfid["data"]["singletcurrent"] = ocorscurm
        corfid["data"]["tripletcurrent"] = ocortcurm
        corfid["data"]["density"] = ocorddm
        corfid["data"]["pair"] = ocorppm
        corfid["data"]["spin"] = ocorssm
        create_group(corfid, "error")
        corfid["error"]["greenfunction"] = ogfuncerr
        corfid["error"]["singletbond"] = ocorsbderr
        corfid["error"]["tripletbond"] = ocortbderr
        corfid["error"]["singletcurrent"] = ocorscurerr
        corfid["error"]["tripletcurrent"] = ocortcurerr
        corfid["error"]["density"] = ocordderr
        corfid["error"]["pair"] = ocorpperr
        corfid["error"]["spin"] = ocorsserr
        close(corfid)

        # data vs path
        EPfile = open( joinpath(datafile,"epath.txt"),"w")
        for i=1:length(oepath)
            @printf(EPfile, "%i, %16.f10, %16.f10, %16.f10, %16.f10\n", i,
                real(oepath[i]),imag(oepath[i]),real(oepatherr[i]),imag(oepatherr[i])
            )
        end
        flush(EPfile)
        close(EPfile)

    else
        nothing
    end
end
