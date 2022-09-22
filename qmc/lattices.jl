# Copyright (c) 2022, Han Xu
# All rights reserved.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree. 


"""Square lattice"""
function MakeSquare()
    nlayer = Row*Col
    xy2siteidx(x::Integer,y::Integer) = (x-1)*Row + y
    # _mattype = ComplexF64
    sitemap = zeros(Integer,Col,Row)
    hopping = zeros(_mattype,nlayer,nlayer)
    bg1 = Array{Tuple{Integer,Integer},1}()
    bg2 = Array{Tuple{Integer,Integer},1}()
    bg3 = Array{Tuple{Integer,Integer},1}()
    bg4 = Array{Tuple{Integer,Integer},1}()
    for x=1:Col
        for y=1:Row
            i = xy2siteidx(x,y)
            sitemap[x,y] = i # store site(x,y) = idx
            xx = x==Col ? 1 : x+1
            yy = y==Row ? 1 : y+1
            # hopping to x+1
            j1 = xy2siteidx(xx,y)
            hopping[j1,i] = -tnn
            hopping[i,j1] = -tnn
            # hopping to y+1
            j2 = xy2siteidx(x,yy)
            hopping[j2,i] = -tnn
            hopping[i,j2] = -tnn
            # bond in 4 group
            if (-1)^(x+y) == 1 # group 1(x),2(y)
                bg1 = i<j1 ? [bg1; (i,j1)] : [bg1; (j1,i)]
                bg2 = i<j2 ? [bg2; (i,j2)] : [bg2; (j2,i)]
            else # group 3(x),4(y)
                bg3 = i<j1 ? [bg3; (i,j1)] : [bg3; (j1,i)]
                bg4 = i<j2 ? [bg4; (i,j2)] : [bg4; (j2,i)]
            end
        end
    end
    bondgroup = [bg1 bg2 bg3 bg4]
    # add flavor N
    kmat = zeros(_mattype,nlayer*N,nlayer*N)
    for a=1:N
        kmat[(a-1)*nlayer+1:a*nlayer, (a-1)*nlayer+1:a*nlayer] = hopping
    end
    # add chemical potiential
    kmat = kmat  -  I * μ
    vals, vecs = eigen(kmat)
    Row>=4 && println("Kmat eigenvals: ", vals)
    return kmat, bondgroup, sitemap, nlayer*N, nlayer
end


"""Square lattice trial wave function for PQMC"""
function MakeSquareTrailWave()
    # make antiboundary lattice
    nlayer = Row*Col
    xy2siteidx(x::Integer,y::Integer) = (x-1)*Row + y
    #_mattype = typeof(ekhalf[1])
    hopping = zeros(_mattype,nlayer,nlayer)
    for x=1:Col
        for y=1:Row
            i = xy2siteidx(x,y)
            # hopping to x+1
            if x==Col # boundary
                xx = 1
                yy = y
                j = xy2siteidx(xx,yy)
                hopping[j,i] = +tnn
                hopping[i,j] = +tnn
            else
                xx = x+1
                yy = y
                j = xy2siteidx(xx,yy)
                hopping[j,i] = -tnn
                hopping[i,j] = -tnn
            end
            # hopping to y+1
            if y==Row # boundary
                xx = x
                yy = 1
                j = xy2siteidx(xx,yy)
                hopping[j,i] = -tnn
                hopping[i,j] = -tnn
            else
                xx = x
                yy = y+1
                j = xy2siteidx(xx,yy)
                hopping[j,i] = -tnn
                hopping[i,j] = -tnn
            end
        end
    end
    # add flavor N
    ktrial = zeros(_mattype,nlayer*N,nlayer*N)
    for a=1:N
        ktrial[(a-1)*nlayer+1:a*nlayer, (a-1)*nlayer+1:a*nlayer] = hopping
    end
    # add chemical potiential
    ktrial = ktrial - I * μ
    # solve the eigenproblem
    vals, vecs = eigen(ktrial)
    Row>=4 && println("Ktrial eigenvals: ", vals)
    # cast the filling number
    fillingnumber = round(Integer, nlayer * N * (0.5+doping)) # sum(vals.<=0) # 
    println("real mu: ", vals[fillingnumber-1:fillingnumber+1])
    println("real doping: ", fillingnumber/nlayer/N-0.5)
    anymat = vecs[:,1:fillingnumber]
    anymatdagger = transpose(conj(anymat))
    # define new slater=exp(-K*Δτ/2)*slater, slaterdag=slaterdag*exp(K*Δτ/2)
    slater = eKHalf * anymat
    slaterdagger = anymatdagger * eMKHalf
    return slater, slaterdagger
end








"""4 site square lattice"""
function Make4Site()
    nlayer = 4
    # _mattype = ComplexF64
    hopping = zeros(_mattype,nlayer,nlayer)
    hopping[1,3] = -tnn
    hopping[3,1] = -tnn
    hopping[1,2] = -tnn
    hopping[2,1] = -tnn
    hopping[2,4] = -tnn
    hopping[4,2] = -tnn
    hopping[3,4] = -tnn
    hopping[4,3] = -tnn
    bg1 = [(1,3)]
    bg2 = [(1,2)]
    bg3 = [(2,4)]
    bg4 = [(3,4)]
    bondgroup = [bg1 bg2 bg3 bg4]
    # add flavor N
    kmat = zeros(_mattype,nlayer*N,nlayer*N)
    for a=1:N
        kmat[(a-1)*nlayer+1:a*nlayer, (a-1)*nlayer+1:a*nlayer] = hopping
    end
    # add chemical potiential
    kmat = kmat  -  I * μ
    sitemap = Array(reshape([1,2,3,4],(2,2))')
    return kmat, bondgroup, sitemap, nlayer*N, nlayer
end


"""4 site square lattice trial wave function"""
function Make4Trial()
    nlayer = 4
    # make antiboundary lattice
    hopping = zeros(_mattype,nlayer,nlayer)
    hopping[1,3] = -tnn
    hopping[3,1] = -tnn
    hopping[1,2] = tnn
    hopping[2,1] = tnn
    hopping[2,4] = tnn
    hopping[4,2] = tnn
    hopping[3,4] = tnn
    hopping[4,3] = tnn
    # add flavor N
    ktrial = zeros(_mattype,nlayer*N,nlayer*N)
    for a=1:N
        ktrial[(a-1)*nlayer+1:a*nlayer, (a-1)*nlayer+1:a*nlayer] = hopping
    end
    # add chemical potiential
    ktrial = ktrial - I * μ
    # solve the eigenproblem
    vals, vecs = eigen(ktrial)
    println("Ktrial eigenvals: ", vals)
    # cast the filling number
    fillingnumber = round(Integer, nlayer * N * (0.5+doping)) # sum(vals.<=0) # nlayer
    println("real mu: ", vals[fillingnumber-1:fillingnumber+1])
    println("real doping: ", fillingnumber/nlayer/N-0.5)
    anymat = vecs[:,1:fillingnumber]
    anymatdagger = Array(anymat')
    # define new slater=exp(-K*Δτ/2)*slater, slaterdag=slaterdag*exp(K*Δτ/2)
    slater = eKHalf * anymat
    slaterdagger = anymatdagger * eMKHalf
    return slater, slaterdagger
end

