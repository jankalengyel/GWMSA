#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

 Functions for the computation of the Geographically Weighted  Multi scale analysis
 on dataset of points carrying (or not) a valued quantity.

    GWMFA.analysis      : performfull GWMFA analysis of a set of points

    GWMFA.localWaveTrans   : compute wavelets and box counting of valued or non valued set of points

    GWMFA.WaveSmoothing : compute various weighted multi resolution quantities from WMFAcount outputs.


@authors: S.G.  Roux  , ENS Lyon, stephane.roux@ens-lyon.fr
          J. Lengyel  , ENS Lyon, janka.lengyel@ens-lyon.fr
          F. Sémécurbe, Insee,    francois.semecurbe@insee.fr

          January 2021. / June 2021
"""
#%% Imports
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from shapely.geometry import Polygon, Point
import scipy.stats as scs

import sys
# %%

def fitMHDcum(Allquant,radius, borne):
    j1=borne[0]
    j2=borne[1]
    AllMom=Allquant[0]
    AllH=Allquant[1]
    AllD=Allquant[2]
    cum1=Allquant[3]
    cum2=Allquant[2]
    zetaq=np.polyfit(np.log(radius[j1:j2]),np.log(AllMom[j1:j2,:]),1)
    Hq=np.polyfit(np.log(radius[j1:j2]),AllH[j1:j2,:],1)
    Dq=np.polyfit(np.log(radius[j1:j2]),AllD[j1:j2,:],1)
    c1=np.polyfit(np.log(radius[j1:j2]),cum1[j1:j2,:],1)
    c2=np.polyfit(np.log(radius[j1:j2]),cum2[j1:j2,:],1)

    return zetaq, Hq, Dq, c1, c2

def MFanalysis(WT,q):
    # %% MF analysis on coeffcient WT

    levels=WT.shape[1]
    #q=np.arange(-5,5.1,.1)
    #q[50:101:10]=[0, 1, 2, 3, 4, 5]
    AllMom=np.zeros((levels,len(q)))
    #AllMomSup=np.zeros((levels,len(q)))
    AllH=np.zeros((levels,len(q)))
    AllD=np.zeros((levels,len(q)))
    Cum1=np.zeros((levels,len(q)))
    Cum2=np.zeros((levels,len(q)))
    Num=np.zeros((levels,))
    for ir in range(levels):
        print(ir,end=' ')
        coef=np.abs(WT[:,ir])
        Num[ir]=coef[coef>0].size
        for iq in range(len(q)):
            #print(iq,end=' ')
            # Moment
            if q[iq]<=0:
                AllMom[ir,iq]=np.mean(coef[coef>0]**q[iq])
            else:
                AllMom[ir,iq]=np.mean(coef**q[iq])

            sumq=np.sum(coef[coef>0]**q[iq])
            Coefhat=coef*0
            Coefhat[coef>0]=coef[coef>0]**q[iq]/sumq
            AllH[ir,iq]=np.sum(Coefhat[coef>0]*np.log(coef[coef>0]))
            AllD[ir,iq]=np.sum(Coefhat[coef>0]*np.log(Coefhat[coef>0]))+np.log(np.prod(coef[coef>0].shape))

        # cumulant
        Cum1[ir,0]=np.mean(np.log(coef[coef>0]))
        Cum2[ir,0]=np.var(np.log(coef[coef>0]))
    print('.')
    return AllMom, AllH, AllD, Cum1, Cum2

# %%
def computehistogram(WT,bins):
    Nr=WT.shape[1]
    hist=np.zeros((bins,Nr))
    centers=np.zeros((bins,Nr))
    lastd=np.zeros(Nr,)
    for ir in range(Nr):
        temp=WT[:,ir]
        temp=temp[np.isfinite(temp)]
        lastd[ir]=np.std(temp)
        # normalize
        temp=temp/np.std(temp)

        htmp, bin_edges = np.histogram(temp, bins=bins)
        centers[:,ir]=(bin_edges[:-1]+bin_edges[1:])/2
        dx=np.mean(np.diff(centers[:,ir]))
        hist[:,ir]=htmp/np.sum(htmp)/dx

    return hist, centers, lastd

# %%
def computeM2FlatnessSkewness(WT,radius):
    #  flatness of FT
    Nr=WT.shape[1]
    Fl=np.ones(Nr,)
    M2=np.ones(Nr,)
    Sk=np.ones(Nr,)
    mean=np.ones(Nr,)
    meanabs=np.ones(Nr,)
    for ir in range(Nr):
        temp=WT[:,ir]
        Fl[ir]=np.mean(temp**4)/np.mean(temp**2)**2/3
        M2[ir]=np.mean(temp**2)
        Sk[ir]=scs.skew(temp)
        mean[ir]=np.mean(temp)
        meanabs[ir]=np.mean(np.abs(temp))

    return M2, Fl, Sk,mean, meanabs

# %%
# %%
def restoGeoPandaFrame(gridpoints, radius ,results, crs = "EPSG:2154"):
    """
     gdf_results = restoGeoPandaFrame(gridpoints, radius ,results)

         return a geopanda dataframe

     input :

         gridpoints - two dimensional array with the grid points position [x,y]
         radius     - one dimensional array with scales (>0)
         results    - two dimensional array of size equal len(gridpoints) X len(radius)

    output :

        out - geopanda dataframe

    ##
     S.G.  Roux, ENS Lyon, December 2020, stephane.roux@ens-lyon.fr
     J.L June 2021

    """

    #  grid dataframe
    df_grid = pd.DataFrame({'x':gridpoints[:,0], 'y':gridpoints[:,1]})
    # get all scales in a single dataframe
    j=0
    mystr = 'R'+radius[j].astype(int).astype('str')
    df_data = pd.DataFrame(results[:,j], columns = [mystr])
    for j in range(1,len(radius)):
        mystr = 'R'+radius[j].astype(int).astype('str')
        df_data.loc[:,mystr] = pd.Series(results[:,j], index=df_data.index)

    gridsize = np.abs(df_grid['x'][0] - df_grid['x'][1])
    gdf_results = gpd.GeoDataFrame( df_data, geometry=[Polygon([(x-gridsize/2, y+gridsize/2), (x+gridsize/2, y+gridsize/2), (x+gridsize/2, y-gridsize/2), (x-gridsize/2, y-gridsize/2), (x-gridsize/2, y+gridsize/2)])
                              for x,y in zip(df_grid.x,df_grid.y)])

    gdf_results.crs = crs


    return gdf_results

# %%
def sumval(ival, xx, r,dist=0):
    """
     out = sumval(i, x)

         return the sum of the value at specified indices.

     input :

         i - array of indices
         x - array of value
         r - value of the current scale (can be usefull)

    output :

        out - sum of the value of array x at indicex i
              out = np.sum(x[i])

    ##
     S.G.  Roux, ENS Lyon, December 2020, stephane.roux@ens-lyon.fr
    """
    dist=np.array(dist)
    if np.prod(dist.shape) ==1 and dist==0: # no distance given

        return np.sum(xx[ival])

    else:
        return np.sum(xx[ival[dist<r]])

# %%
def meanval(ival, xx, r,dist=0):
    """
      out = meanval(i, x, r, dist)

          return the mean of the value at specified indices
          with condition dist<r.

      input :

          i - array of indices
          x - array of value
          r - value of the current scale (can be usefull)
          dist - distance of the point i

    output :

        out - mean of the value of array x at indicex i
              out = np.mean(x[i])

    ##
      S.G.  Roux, ENS Lyon, December 2020, stephane.roux@ens-lyon.fr
      J.L. June 2021
    """
    dist=np.array(dist)
    if np.prod(dist.shape) ==1 and dist==0: # no distance given

        return np.mean(xx[ival])
    else:

        return np.mean(xx[ival[dist<r]])


# %%
def stdval(ival, xx, r,dist=0):
    """
     out = stdval(i, x)

         return the standard deviation of the value at specified indices.

     input :

         i - array of indices
         x - array of value
         r - value of the current scale (can be usefull)

    output :

        out - standard deviation of the value of array x at indicex i
              out = np.std(x[i])

    ##
     S.G.  Roux, ENS Lyon, December 2020, stephane.roux@ens-lyon.fr
     J.L. June 2021
    """
    dist=np.array(dist)
    if np.prod(dist.shape) ==1 and dist==0: # no distance given

        return np.std(xx[ival])
    else:

        return np.std(xx[ival[dist<r]])
# %%
def geographicalWeighting(Count,ii, T,dd):

    # Weighting function
    W = (T**2-dd**2)**2/T**4
    W=W/np.sum(W,0) # normalization

    Wfinal=np.tile(W, (Count.shape[1], 1)).T

    tmp_fixed = Count[ii,:]
    wtmp_fixed = Wfinal * tmp_fixed

    return wtmp_fixed

# %%
def geographicalWeight(dd,T,Nr):

    # Weighting function
    W = (T**2-dd**2)**2/T**4
    W=W/np.sum(W,0) # normalization

    Wfinal=np.tile(W, (Nr, 1)).T


    return Wfinal

# %%
def localWaveTrans(data, radius, T=0, Nanalyse=2**16, function = 0, glob = 0):
    """

   [Count, Wave] = localWaveTrans(data, radius, Nanalyse=2**16, function = 0, glob = 0))


   Compute box-counting and wavelet coefficient on a valued/non valued set of data points.

   If the data are not valued, count for every data point
           -- the number of neighboors in ball of radius r (GWFA).
           -- the wavelet coeficient at scale r (GWMFA).

   If the data are valued, count for every datapoint
           -- the number of neighboors in ball of radius r and mean/sum/std the value of
              the selected points (valued analysis).
           -- the wavelet coeficient at scale r on the value of the selected points (valued analysis).

   Input :

       data     - matrix of size Nx2 --> position (x,y) for N points
                  matrix of size Nx3 --> position (x,y,value) for N points
       radius   - list of scales to be investigated
       T        - local environment distance
       Nanalyse - number of points to analyse in one bach. Default is 2**16 points.
                  If Nanalyse=0, compute all the points in once (dangerous!!!)
       function - function used for the valued analysis : this function is called for
                  evaluating the count at scale r (default 0: mean of the value, 1: sum, 2: std)
       glob     - optional normalisation: values at each r weighted by the value of the T weighing window of WaveSmoothing (default non-weighted)

   Output :

       Count    - matrix of size Nxlength(radius) with box-counting
       Wave     - matrix of size Nxlength(radius) with wavelet coefficients


   Usage exemple :


    ##
     S.G.  Roux, ENS Lyon, December 2020,  stephane.roux@ens-lyon.fr
     J.L. June 2021
    """
    #print(function)
    # check input data
    si=data.shape
    if si[1] < 2 or si[1]>3:
        raise TypeError('The second dimension of first argument must be of length 2 or 3.')

    isvalued=0 # non valued analysis
    if si[1] > 2: # valued analysis
        isvalued=1
        print('Valued data analysis.')
        val=data[:,2]
        data=data[:,0:2]
        if function =='mean':
            func=meanval
        elif function =='sum':
             func=sumval
        elif function =='std':
             func=stdval
        else:
             raise TypeError('Parameterfunction can be "sum", "mean" or "std".')

    else:
        val=data[:,0]*0+1
        print('Non valued data analysis.')

    radius=np.array(radius)
    # if T is present gives for each point N(T)
    #  don't do the normalisation
    # if T=0 do not do that
    if T>0 and radius[radius>T].shape[0] >0:
        raise TypeError('The last argument must be greater than a sqrt(2)*radius.')
        # print('The last argument must be greater than a sqrt(2)*radius.')
    else:
        scalemax=max(np.max(np.sqrt(2)*radius),T)

    if T==0 or glob==0: # beter a small scalemax
        scalemax=np.max(np.sqrt(2)*radius)


    N = data.shape[0]
    # sub parameters to cut the data in bunch
    if Nanalyse==0:
        Nanalyse=N
        Nbunch=1
    else:
        # find how many bunchs
        Nbunch=np.int(np.ceil(N/Nanalyse))

    ## Count the number of neighbours of each point e within each radius r  ##
    Count=np.zeros((N,radius.shape[0]),dtype=np.float)
    Count2=np.zeros((N,radius.shape[0]),dtype=np.float)
    CountG=np.zeros((N,),dtype=np.float)
    # Countbis=np.zeros((N,radius.shape[0]),dtype=np.float)
    # Count2bis=np.zeros((N,radius.shape[0]),dtype=np.float)
    # CountGbis=np.zeros((N,),dtype=np.float)

    for ibunch in range(Nbunch):
        print('bunch {:d}/{:d} : '.format(ibunch+1,Nbunch), end=' ')
        sys.stdout.flush()

        # get data index of current bunch
        index=np.arange(ibunch*Nanalyse,(ibunch+1)*Nanalyse,1)
        index=index[index<N]

        # we restrict the tree to points whithin a radius T (which must be > radius2)
        mi=np.min(data[index,:], axis=0)
        ma=np.max(data[index,:], axis=0)
        IndexNear=np.where((data[:,0] >mi[0]-T) & (data[:,0] <ma[0]+T) & (data[:,1]  >mi[1]-T) & (data[:,1] <ma[1]+T))[0]

        # make the tree
        treetot = KDTree(data[IndexNear,:])

        # valued analysis
        if isvalued:

            IdxTot, Disttot =treetot.query_radius(data[index,0:2], r = scalemax,count_only=False,return_distance=True)
            #func2 = meanval2
            if (function == meanval) and glob ==1:
                raise TypeError('No option "glob" for mean valued analysis')


            for i in range(radius.size):
                print(i, end=' ')
                sys.stdout.flush()
                Count2[index,-1-i]  = np.array([ func(IndexNear[ii],val, np.sqrt(2)*radius[-1-i],dd) for ii, dd in zip(IdxTot,Disttot)])
                Count[index,-1-i]  = np.array([ func(IndexNear[ii],val, radius[-1-i],dd) for ii, dd in zip(IdxTot,Disttot)])

            if glob==1: # glob =1 -->normalisation by vfunc(T)
                CountG[index] = np.array([ func(IndexNear[ii],val, T,dd) for ii, dd in zip(IdxTot,Disttot)])

            print('')

        else:  # non valued analysis
            _ , Disttot = treetot.query_radius(data[index,0:2], r = scalemax,count_only=False,return_distance=True)


            for i in range(radius.size):
                print(i, end=' ')
                sys.stdout.flush()
                Count2[index,-1-i] = np.array([  ii[ii<np.sqrt(2)*radius[-1-i]].shape[0] for ii in Disttot ])
                Count[index,-1-i]  = np.array([  ii[ii<radius[-1-i]].shape[0] for ii in Disttot ])


            # if glob == 0:
            #     for i in range(radius.size):
            #         print(i, end=' ')
            #         tree = KDTree(data[IndexNear,:])
            #         Idx = tree.query_radius(data[index,0:2], r = np.sqrt(2)*radius[-1-i],count_only=False,return_distance=False)
            #         Count2bis[index,-1-i]  = np.array([ ii.shape[0] for ii in Idx ])
            #         #Count2[index,-1-i] = np.array([  ii[ii<np.sqrt(2)*radius[-1-i]].shape[0] for ii in Disttot ])

            #         uu=np.array([ ii.shape[0] for ii in Idx ])
            #         vv=np.array([  ii[ii<np.sqrt(2)*radius[-1-i]].shape[0] for ii in Disttot ])

            #         print(np.allclose(uu,vv))
            #         sys.stdout.flush()
            #         IndexNear = IndexNear[np.unique(np.concatenate(Idx))]

            #         tree = KDTree(data[IndexNear,:])
            #         Idx = tree.query_radius(data[index,0:2], r = radius[-1-i],count_only=False,return_distance=False)
            #         Countbis[index,-1-i]  = np.array([ ii.shape[0] for ii in Idx ])

            #         IndexNear = IndexNear[np.unique(np.concatenate(Idx))]


            # print(np.allclose(Count2,Count2bis))

            if glob == 1:
                CountG[index] = np.array([  ii[ii<T].shape[0] for ii in Disttot ])

            # if glob == 1:
            #     for i in range(radius.size):
            #         print(i, end=' ')
            #         tree = KDTree(data[IndexNear,:])
            #         CountGlobal = tree.query_radius(data[index,0:2], r = T,count_only=False,return_distance=False)
            #         Idx = tree.query_radius(data[index,0:2], r = np.sqrt(2)*radius[-1-i],count_only=False,return_distance=False)
            #         Count2[index,-1-i]  = np.array([ ii.shape[0] for ii in Idx ])/np.array([ ii.shape[0] for ii in CountGlobal ])
            #         IndexNear = IndexNear[np.unique(np.concatenate(Idx))]

            #         CountGbis[index]=np.array([ ii.shape[0] for ii in CountGlobal ])

            #         tree = KDTree(data[IndexNear,:])
            #         CountGlobal = tree.query_radius(data[index,0:2], r = T,count_only=False,return_distance=False)
            #         Idx = tree.query_radius(data[index,0:2], r = radius[-1-i],count_only=False,return_distance=False)
            #         Count[index,-1-i]  = np.array([ ii.shape[0] for ii in Idx ])/np.array([ ii.shape[0] for ii in CountGlobal ])
            #         IndexNear = IndexNear[np.unique(np.concatenate(Idx))]

            #print(np.allclose(CountG,CountGbis))
            print('')

    return 2*Count-Count2, Count, CountG
# %%
def computeGridValue(tree,Count,IndexNear,index,mygrid, radius, T,kernel,ismom,k):

    if kernel == 0:
            neighbors_i_fixed, dist_ie_fixed = tree.query_radius(mygrid[index,:], r = T, count_only=False, return_distance=True, sort_results=False)

    else:
        neighbors_i_fixed, dist_ie_fixed = tree.query_radius(mygrid[index,:], r = T, count_only=False, return_distance=True, sort_results=False)
        dist_ie_adaptive, neighbors_i_adaptive = tree.query(mygrid[index,:], k = k, return_distance=True, sort_results=False)
        # convert in an array of array (same as fixed)
        #dist_ie_adaptive=list(dist_ie_adaptive)
        #neighbors_i_adaptive=list(neighbors_i_adaptive)

        Tloc=np.array([ np.max(dist_ie_adaptive[igrid])  for igrid in range (len(index))])

        # Test if  adaptative environment > radius max
        le = Tloc[Tloc<np.sqrt(2)*radius.max()].shape[0]
        if le >1:
            txt="Local environment is smaller than radius max for {:.2f}% points--> choose larger k".format(le/Tloc.shape[0]*100)
            print(txt)
            #raise TypeError(txt)
        else :
            print('Adaptative Locals Environment :',end=' ')
            print('Mean size {:.2f}, std {:.2f}'.format(np.mean(Tloc),np.std(Tloc)))
            sys.stdout.flush()

        # adaptative environment between radius max and T
        # correct distances and index of neighbors
        if Tloc[Tloc<T].size > 0:
            lesi, =np.where(Tloc<T)
             #print(len(dist_ie_adaptive[lesi[1]]), len(dist_ie_fixed[lesi[1]]))

            dist_ie_fixed[lesi]=[dist_ie_adaptive[lesi[ii]] for ii in range(len(lesi))]
            #print(len(dist_ie_adaptive[lesi[1]]), len(dist_ie_fixed[lesi[1]]))
            neighbors_i_fixed[lesi]=[neighbors_i_adaptive[lesi[ii]] for ii in range(len(lesi))]
            Tloc[Tloc>T]=T

    print('.',end='')
    sys.stdout.flush()


    wtmp_fixed2=[ geographicalWeighting(Count,IndexNear[neighbors_i_fixed[igrid]],T, dist_ie_fixed[igrid]) for igrid in range (len(index))]
    tmp_fixed2=[ Count[IndexNear[neighbors_i_fixed[igrid]],:] for igrid in range (len(index))]
    ltmp=np.array([ len(dist_ie_fixed[igrid]) for igrid in range (len(index))])
    Wfinal2=[  geographicalWeight(dist_ie_fixed[igrid], T,len(radius)) for igrid in range (len(index))]

    if ismom: # compute moments
        Mom0 = np.array([np.sum( tmp_fixed2[igrid]**0  , axis=0)  for igrid in range(len(index))])
        print('.',end='')
        Mom1 = np.array([np.sum( np.abs(wtmp_fixed2[igrid]) , axis=0)  for igrid in range(len(index))])
        print('.',end='')
        Mom2 = np.array([np.sum( np.abs(wtmp_fixed2[igrid] * tmp_fixed2[igrid]), axis=0)  for igrid in range(len(index))])
        print('.',end='')
        Mom3 = np.array([np.sum( np.abs(wtmp_fixed2[igrid] * tmp_fixed2[igrid]**2), axis=0)  for igrid in range(len(index))])
        print('.',end='')
        Mom4 = np.array([np.sum( np.abs(wtmp_fixed2[igrid] * tmp_fixed2[igrid]**3), axis=0)  for igrid in range(len(index))])
        print('.',end='')
        sys.stdout.flush()
        Mom=np.stack([Mom0, Mom1, Mom2, Mom3, Mom4],axis=0)
    else:
        Mom=[]

    # Non weighted std
    NWstd = np.array([np.std( tmp_fixed2[igrid], axis=0, ddof=1) for igrid in range(len(index))])
    NWstd[ltmp<=10,:] =0
    NWmean = np.array([np.mean( tmp_fixed2[igrid], axis=0) for igrid in range(len(index))])
    NWmean[ltmp<=10,:]=0
    print('.',end='')
    sys.stdout.flush()

    # weighted mean and std
    #Wfinal2=[  geographicalWeight(dist_ie_fixed[igrid], T,len(radius)) for igrid in range (len(index))]
    average = np.array([np.average(tmp_fixed2[igrid], weights=Wfinal2[igrid], axis=0) for igrid in range(len(index))])
    Wmean = average
    Wmean[ltmp<=10,:]=0
    Wstd  = np.array([np.average((tmp_fixed2[igrid]-average[igrid])**2, weights=Wfinal2[igrid], axis=0) for igrid in range(len(index))])
    Wstd[ltmp<=10,:]=0
    Wstd = np.sqrt(Wstd)
    print(' ',end='')
    sys.stdout.flush()

    return NWmean, NWstd, Wmean, Wstd, Mom
# %%
def WaveSmoothingOptim(data,Count,X,Y,radius, T,  Nanalyse=2**16, k = 0, ismom=0, kernel = 0):
    """

   res = WaveSmoothing(data,Wave,X,Y,radius,T,Nanalyse=2**16, k = 0, ismom=0, kernel = 0))


   Compute kernel smoothing of the wavelet coefficient of a dataset of points.
   The geographical weighting is obtained using the bi-squared function of size T.
   The wavelet coeficients are obtained using the function GWMFA_count.m

   Input :

       data    - matrix of size Nx2 --> position (x,y) for N points
                 matrix of size Nx3 --> position (x,y,value) for N points
       Wave    - matrix of size Nxlength(radius) with wavelet count
                 oCan be obtained using the function  GWFA_count.m
       X       - array of dim 2 with x-postion of the grid nodes
       Y       - array of dim 2 with y-postion of the grid nodes : X and Y must have the same size
       radius  - list of scales to be investigated
       T       - bandwith of fixed kernel OR distance upper boundary of adaptive kernel
       k       - number of min neighbors for the adaptive kernel smoothing
       kernel  - if equals to 1 compute with adaptive kernel (default 0 which is fixed).
                 L_min = sqrt(2)*r_max, L_max = T, in between adaptive knn neighbors
      Nanalyse - number of points to analyse in one bach. Default is 2**16 points.
       isMom   - if equals to 1 compute also moment of order À to 4. Default 0.


   Output :

       res - list of two dimensional numpy matrix (of size length(data) X length(radius)

              res[Ø] = Wmean  : weighted mean
              res[1] = Wstd    : weighted standert deviation
              res[2] = NWratio : non weighted mean
              res[3] = NWstd   : non weighted stabdard deviation

            if isMom=1
              res[4] = Mom : weighted moment (order 0 to 4) of the absolute value
                      of thecoefficient. matrix of size 5 X length(data) X length(radius)



    ##
     S.G.  Roux, ENS Lyon, December 2020,  stephane.roux@ens-lyon.fr
     J.L June 2021
    """


    si=data.shape
    if si[1] < 2 or si[1]>3:
        raise TypeError('The second dimension of first argument must be of length 2 or 3.')

    if Count.ndim != 2:
        raise TypeError('The second arument must be two dimensional.')

    if Count.shape[0] != si[0]:
        raise TypeError('The twofirst arument must have the same length.')

    if radius[radius>T].shape[0] >0:
        raise TypeError('The last argument must be greater than a sqrt(2)*radius.')

    # check grid
    if np.sum(np.abs(np.array(X.shape)-np.array(Y.shape)))>0:
        raise TypeError('X and Y must have the same size.')

    # create grid points
    mygrid=np.column_stack([X.flatten(),Y.flatten()])
    gridlength=mygrid.shape[0]
    gridsizecurr = X[0][1] - X[0][0]

    if Nanalyse==0:
        Nanalyse=gridlength
        Nbunch=1
    else:
        # find how many bunchs
        Nbunch=np.int(np.ceil(gridlength/Nanalyse))

    # results allocations
    if ismom:
        # mean wave
        #Mom = np.zeros( (6, gridlength,radius.size), dtype=np.float)
        Mom = np.zeros( (5, gridlength,radius.size), dtype=np.float)

    if k:
        k = k
    # NWstd wave
    NWstd = np.zeros( (gridlength,radius.size), dtype=np.float)
    # Wstd spatialized wave
    Wstd = np.zeros( (gridlength,radius.size), dtype=np.float)

    NWmean= np.zeros( (gridlength,radius.size), dtype=np.float)
    # Wstd spatialized wave
    Wmean = np.zeros( (gridlength,radius.size), dtype=np.float)

    # the 3 following are now computed outside the function
    # ratio wave
    #NWratio = np.zeros( (gridlength,radius.size), dtype=np.float)
    #Wratio  = np.zeros( (gridlength,radius.size), dtype=np.float)
    # inverse ratio wave
    # Wlacun  = np.zeros( (gridlength,radius.size), dtype=np.float)

    # for debug
    # Mom2 = np.zeros( (6, gridlength,radius.size), dtype=np.float)
    # NWstd2 = np.zeros( (gridlength,radius.size), dtype=np.float)
    # Wstd2 = np.zeros( (gridlength,radius.size), dtype=np.float)
    # NWmean2= np.zeros( (gridlength,radius.size), dtype=np.float)
    # Wmean2 = np.zeros( (gridlength,radius.size), dtype=np.float)
    # NWratio2 = np.zeros( (gridlength,radius.size), dtype=np.float)
    # Wratio2  = np.zeros( (gridlength,radius.size), dtype=np.float)

    # # inverse ratio wave
    # #NWlacun  = np.zeros( (gridlength,radius.size), dtype=np.float)
    # Wlacun2  = np.zeros( (gridlength,radius.size), dtype=np.float)

    # Loop on bunch
    for ibunch in range(Nbunch):
        # %
        print('bunch {:d}/{:d} '.format(ibunch+1,Nbunch), end=' ')
        sys.stdout.flush()
        # get data index of current bunch
        index=np.arange(ibunch*Nanalyse,(ibunch+1)*Nanalyse,1)
        index=index[index<gridlength]

        # we restrict the tree to points whithin a radius T (which must be > radius2)
        mi=np.min(mygrid[index,:], axis=0)
        ma=np.max(mygrid[index,:], axis=0)
        IndexNear=np.where((data[:,0] >mi[0]-T) & (data[:,0] <ma[0]+T) & (data[:,1]  >mi[1]-T) & (data[:,1] <ma[1]+T))[0]

        #if is_empty = False:
        tree = KDTree(data[IndexNear,0:2])

        # only compute value for grid points that contain raw data - eliminates artefacts at small T
        thelengths = tree.query_radius(mygrid[index,:], r = np.sqrt(2)*gridsizecurr, count_only=True, return_distance=False)
        IdxMin, = np.where(thelengths>0.)

        index = index[IdxMin]
        thelengths=thelengths[IdxMin]

        # begin modif
        #lengrid[index]=IdxMin
        Nanalyse2=300000
        cumsumbunch=np.cumsum(thelengths)
        Nflowers=int(np.ceil(np.sum(thelengths)/Nanalyse2))
        print(Nflowers,np.sum(thelengths),Nanalyse2)
        for iflower in range(Nflowers):
            i1,=np.where(iflower*Nanalyse2 <= cumsumbunch)
            i2,=np.where(cumsumbunch <  (iflower+1)*Nanalyse2)
            flowers=np.intersect1d(i1,i2)
            if ismom:
                NWmean[index[flowers],:], NWstd[index[flowers],:], Wmean[index[flowers],:], Wstd[index[flowers],:], Mom[:,index[flowers],:]=computeGridValue(tree,Count,IndexNear,index[flowers],mygrid, radius,T, kernel,ismom,k)
            else:
                NWmean[index[flowers],:], NWstd[index[flowers],:], Wmean[index[flowers],:], Wstd[index[flowers],:], Mom =computeGridValue(tree,Count,IndexNear,index[flowers],mygrid, radius,T, kernel,ismom,k)

        print('.')
        # end modif

        # following is in the function --> comment
        # if kernel == 0:
        #     neighbors_i_fixed, dist_ie_fixed = tree.query_radius(mygrid[index,:], r = T, count_only=False, return_distance=True, sort_results=False)

        # else:
        #     neighbors_i_fixed, dist_ie_fixed = tree.query_radius(mygrid[index,:], r = T, count_only=False, return_distance=True, sort_results=False)
        #     dist_ie_adaptive, neighbors_i_adaptive = tree.query(mygrid[index,:], k = k, return_distance=True, sort_results=False)
        #     # convert in an array of array (same as fixed)
        #     #dist_ie_adaptive=list(dist_ie_adaptive)
        #     #neighbors_i_adaptive=list(neighbors_i_adaptive)

        #     Tloc=np.array([ np.max(dist_ie_adaptive[igrid])  for igrid in range (len(index))])

        #     # Test if  adaptative environment > radius max
        #     le = Tloc[Tloc<np.sqrt(2)*radius.max()].shape[0]
        #     if le >1:
        #         txt="Local environment is smaller than radius max for {:.2f}% points--> choose larger k".format(le/Tloc.shape[0]*100)
        #         print(txt)
        #         #raise TypeError(txt)
        #     else :
        #         print('Adaptative Locals Environment :',end=' ')
        #         print('Mean size {:.2f}, std {:.2f}'.format(np.mean(Tloc),np.std(Tloc)))
        #         sys.stdout.flush()

        #     # adaptative environment between radius max and T
        #     # correct distances and index of neighbors
        #     if Tloc[Tloc<T].size > 0:
        #         lesi, =np.where(Tloc<T)
        #          #print(len(dist_ie_adaptive[lesi[1]]), len(dist_ie_fixed[lesi[1]]))

        #         dist_ie_fixed[lesi]=[dist_ie_adaptive[lesi[ii]] for ii in range(len(lesi))]
        #         #print(len(dist_ie_adaptive[lesi[1]]), len(dist_ie_fixed[lesi[1]]))
        #         neighbors_i_fixed[lesi]=[neighbors_i_adaptive[lesi[ii]] for ii in range(len(lesi))]
        #         Tloc[Tloc>T]=T

        # print('.',end='')
        # sys.stdout.flush()
        # # for debug
        # #start = time.process_time()

        # wtmp_fixed2=[ geographicalWeighting(Count,IndexNear[neighbors_i_fixed[igrid]],T, dist_ie_fixed[igrid]) for igrid in range (len(index))]
        # tmp_fixed2=[ Count[IndexNear[neighbors_i_fixed[igrid]],:] for igrid in range (len(index))]
        # ltmp=np.array([ len(dist_ie_fixed[igrid]) for igrid in range (len(index))])
        # Wfinal2=[  geographicalWeight(dist_ie_fixed[igrid], T,len(radius)) for igrid in range (len(index))]

        # if ismom: # compute moments
        #     Mom[0,index,:] = np.array([np.sum( tmp_fixed2[igrid]**0  , axis=0)  for igrid in range(len(index))])
        #     print('.',end='')
        #     Mom[1,index,:] = np.array([np.sum( np.abs(wtmp_fixed2[igrid]) , axis=0)  for igrid in range(len(index))])
        #     print('.',end='')
        #     Mom[2,index,:] = np.array([np.sum( np.abs(wtmp_fixed2[igrid] * tmp_fixed2[igrid]), axis=0)  for igrid in range(len(index))])
        #     print('.',end='')
        #     Mom[3,index,:] = np.array([np.sum( np.abs(wtmp_fixed2[igrid] * tmp_fixed2[igrid]**2), axis=0)  for igrid in range(len(index))])
        #     print('.',end='')
        #     Mom[4,index,:] = np.array([np.sum( np.abs(wtmp_fixed2[igrid] * tmp_fixed2[igrid]**3), axis=0)  for igrid in range(len(index))])
        #     print('.',end='')
        #     sys.stdout.flush()

        # # Non weighted std
        # NWstd[index,:] = np.array([np.std( tmp_fixed2[igrid], axis=0, ddof=1) for igrid in range(len(index))])
        # NWstd[index[ltmp<=10],:] =0
        # NWmean[index,:] = np.array([np.mean( tmp_fixed2[igrid], axis=0) for igrid in range(len(index))])
        # print('.',end='')
        # sys.stdout.flush()

        # # weighted mean and std
        # #Wfinal2=[  geographicalWeight(dist_ie_fixed[igrid], T,len(radius)) for igrid in range (len(index))]
        # average = np.array([np.average(tmp_fixed2[igrid], weights=Wfinal2[igrid], axis=0) for igrid in range(len(index))])
        # Wmean[index,:] = average
        # Wmean[index[ltmp<=10],:]=0
        # Wstd[index,:]  = np.array([np.average((tmp_fixed2[igrid]-average[igrid])**2, weights=Wfinal2[igrid], axis=0) for igrid in range(len(index))])
        # Wstd[index[ltmp<=10],:]=0
        # print('.')
        # sys.stdout.flush()

        # en done in fnction

        # the 3 following can be computed outside the function
        # Non weighted ratio
        #NWratio=np.divide(NWmean,NWstd, out=np.zeros_like(NWstd), where=NWstd!=0)

        # weighted ratio
        #Wratio=np.divide(Wmean,Wstd, out=np.zeros_like(Wstd), where=Wstd!=0)

        # weighted lacunarity@
        #Wlacun = np.divide(Wstd,Wmean, out=np.zeros_like(Wstd), where=Wmean!=0)**2



        # for debug
        #print(time.process_time() - start)

        # for debug
        # start = time.process_time()
    #     # old method
    #     if kernel == 0:
    #         neighbors_i_fixed, dist_ie_fixed = tree.query_radius(mygrid[index,:], r = T, count_only=False, return_distance=True, sort_results=False)


    #         print('.')
    #     else:
    #         neighbors_i_fixed, dist_ie_fixed = tree.query_radius(mygrid[index,:], r = T, count_only=False, return_distance=True, sort_results=False)
    #         dist_ie_adaptive, neighbors_i_adaptive = tree.query(mygrid[index,:], k = k, return_distance=True, sort_results=False)

    #     for igrid in range (len(index)):

    #         # distance to current grid point
    #         dtmp_fixed = dist_ie_fixed[igrid]
    #         # coefficient of the grid point neighbors
    #         tmp_fixed = Count[IndexNear[neighbors_i_fixed[igrid]]]
    #         #print(np.allclose(tmp_fixed,tmp_fixed2[igrid]))

    #         if kernel == 0:
    #             if len(dtmp_fixed) != 0:

    #                 # Weights original
    #                 W = (T**2-dtmp_fixed**2)**2/T**4
    #                 W=W/np.sum(W,0) # normalization

    #                 Wfinal=np.tile(W, (radius.shape[0], 1)).T
    #                 wtmp_fixed = Wfinal * tmp_fixed

    #                 #uu=geographicalWeighting(Count,IndexNear[neighbors_i_fixed[igrid]],T, dist_ie_fixed[igrid])
    #                 #np.allclose(wtmp_fixed,uu)
    #                 #print(np.allclose(wtmp_fixed,wtmp_fixed2[igrid]))

    #                 if ismom: # compute moments
    #                     Mom2[0,index[igrid],:] = np.sum( tmp_fixed**0       , axis=0)
    #                     Mom2[1,index[igrid],:] = np.sum( wtmp_fixed , axis=0)
    #                     Mom2[2,index[igrid],:] = np.sum( wtmp_fixed * tmp_fixed, axis=0)
    #                     Mom2[3,index[igrid],:] = np.sum( wtmp_fixed * tmp_fixed**2, axis=0)
    #                     Mom2[4,index[igrid],:] = np.sum( wtmp_fixed * tmp_fixed**3, axis=0)
    #                     # Flatness
    #                     Mom2[5,index[igrid],:] = np.sum(np.abs(tmp_fixed)**4, axis=0)/np.sum(np.abs(tmp_fixed)**2, axis=0)/3
    #                     #print('OK')
    #                     #print(np.mean(wtmp_fixed, axis=0),np.mean(tmp_fixed, axis=0),Mom[1,index[igrid],:] )

    #                 if  len(dtmp_fixed) >10: # compute ratio
    #                     # Non weighted std
    #                     NWstd2[index[igrid]] = np.std( tmp_fixed, axis=0, ddof=1)
    #                     NWmean2[index[igrid]] = np.mean( tmp_fixed, axis=0)
    #                     # Non weighted dispersion index: 0.1 due to the edge problem

    #                     # weighted std
    #                     #Wstd[index[igrid]]  = np.std( wtmp_fixed, axis=0, ddof=1)
    #                     #Wmean[index[igrid]] = np.mean( wtmp_fixed, axis=0)

    #                     average = np.average(tmp_fixed, weights=Wfinal, axis=0)
    #                     Wmean2[index[igrid]] = average
    #                     Wstd2[index[igrid]]  = np.average((tmp_fixed-average)**2, weights=Wfinal, axis=0)
    #                     #print(np.allclose(Wmean[index[igrid]],Wmean2[index[igrid]]))

    #                     # Non weighted ratio
    #                     stdtmp_fixed=np.tile(NWstd2[index[igrid]],(tmp_fixed.shape[0], 1))
    #                     ttmp_fixed = np.divide(tmp_fixed, stdtmp_fixed , out=np.zeros_like(tmp_fixed), where=stdtmp_fixed!=0)
    #                     #ttmp_fixed = np.sum(Wfinal * ttmp_fixed,axis=0) # one error here
    #                     ttmp_fixed = np.mean(ttmp_fixed,axis=0)
    #                     NWratio2[index[igrid]] = ttmp_fixed
    #                     #print(np.allclose(NWratio[index[igrid]],NWratio2[index[igrid]]))

    #                     # weighted ratio
    #                     stdtmp_fixed = np.tile(Wstd2[index[igrid]],(tmp_fixed.shape[0], 1))
    #                     ttmp_fixed = np.divide(tmp_fixed , stdtmp_fixed , out=np.zeros_like(tmp_fixed), where=stdtmp_fixed!=0)
    #                     ttmp_fixed = np.sum(Wfinal * ttmp_fixed,axis=0)
    #                     Wratio2[index[igrid],:] = ttmp_fixed

    #                     # weighted lacunarity
    #                     #print(np.max(Wstd2[index[igrid]]-Wstd[index[igrid]]))
    #                     #print(np.max(Wmean2[index[igrid]]-Wmean2[index[igrid]]))

    #                     stdtmp_fixed =  np.tile(Wstd2[index[igrid]],(tmp_fixed.shape[0], 1))
    #                     meantmp_fixed = np.tile(Wmean2[index[igrid]],(tmp_fixed.shape[0], 1))
    #                     ttmp_fixed = (np.divide(stdtmp_fixed , meantmp_fixed , out=np.zeros_like(tmp_fixed), where=meantmp_fixed!=0))**2
    #                     #ttmp_fixed[0,:]-(Wstd[index[igrid]]/Wmean[index[igrid]])**2

    #                     ttmp_fixed = np.sum(Wfinal * ttmp_fixed,axis=0)
    #                     Wlacun2[index[igrid],:] = ttmp_fixed

    #         else:
    #             if len(dtmp_fixed != 0):
    #                 dtmp_adaptive= dist_ie_adaptive[igrid]
    #                 # coefficient of the grid point neighbors
    #                 tmp_adaptive = Count[IndexNear[neighbors_i_adaptive[igrid]]]

    #                 # Weights =
    #                 TLoc = np.max(dtmp_adaptive)
    #                 if TLoc < np.sqrt(2)*radius.max():
    #                     #break
    #                     print("Local environment is smaller than radius max --> choose larger k")

    #                     #raise TypeError("Local environment is smaller than radius max --> choose larger k")
    #                 elif TLoc > T:

    #                     # Weights original
    #                     W = (T**2-dtmp_fixed**2)**2/T**4
    #                     W=W/np.sum(W,0) # normalization

    #                     Wfinal=np.tile(W, (radius.shape[0], 1)).T
    #                     wtmp_fixed = Wfinal * tmp_fixed


    #                     if ismom: # compute moments
    #                         Mom[0,index[igrid],:] = np.sum( tmp_fixed**0       , axis=0)
    #                         Mom[1,index[igrid],:] = np.sum( wtmp_fixed , axis=0)
    #                         Mom[2,index[igrid],:] = np.sum( wtmp_fixed * tmp_fixed, axis=0)
    #                         Mom[3,index[igrid],:] = np.sum( wtmp_fixed * tmp_fixed**2, axis=0)
    #                         Mom[4,index[igrid],:] = np.sum( wtmp_fixed * tmp_fixed**3, axis=0)
    #                         #print('grr1')


    #                     if  len(dtmp_fixed) >0: # compute ratio
    #                         # Non weighted std
    #                         NWstd[index[igrid]] = np.std( tmp_fixed, axis=0, ddof=1)
    #                         # Non weighted dispersion index: 0.1 due to the edge problem
    #                         # weighted std
    #                         #Wstd[index[igrid]]  = np.std( wtmp_fixed, axis=0, ddof=1)
    #                         #Wmean[index[igrid]] = np.mean( wtmp_fixed, axis=0)

    #                         # corrected: verify
    #                         average = np.average(tmp_fixed, weights=Wfinal, axis=0)
    #                         Wmean[index[igrid]] =average
    #                         Wstd[index[igrid]]  = np.average((tmp_fixed-average)**2, weights=Wfinal,axis=0)

    #                         # Non weighted ratio
    #                         stdtmp_fixed=np.tile(NWstd[index[igrid]],(tmp_fixed.shape[0], 1))
    #                         ttmp_fixed = np.divide(tmp_fixed, stdtmp_fixed , out=np.zeros_like(tmp_fixed), where=stdtmp_fixed!=0)
    #                         ttmp_fixed = np.sum(Wfinal * ttmp_fixed,axis=0)
    #                         NWratio[index[igrid]] = ttmp_fixed # np.sum( Wfinal * ttmp,axis=0)
    #                         # weighted ratio
    #                         stdtmp_fixed = np.tile(Wstd[index[igrid]],(tmp_fixed.shape[0], 1))
    #                         ttmp_fixed = np.divide(tmp_fixed , stdtmp_fixed , out=np.zeros_like(tmp_fixed), where=stdtmp_fixed!=0)
    #                         ttmp_fixed = np.sum(Wfinal * ttmp_fixed,axis=0)
    #                         Wratio[index[igrid],:] = ttmp_fixed #np.sum( Wfinal * ttmp,axis=0)
    #                         # weighted lacunarity
    #                         stdtmp_fixed = np.tile(Wstd[index[igrid]],(tmp_fixed.shape[0], 1))
    #                         meantmp_fixed = np.tile(Wmean[index[igrid]],(tmp_fixed.shape[0], 1))
    #                         ttmp_fixed = (np.divide(stdtmp_fixed , meantmp_fixed , out=np.zeros_like(tmp_fixed), where=meantmp_fixed!=0))**2
    #                         ttmp_fixed = np.sum(Wfinal * ttmp_fixed,axis=0)
    #                         Wlacun[index[igrid],:] = ttmp_fixed #np.sum( Wfinal * ttmp,axis=0)

    #                 else:

    #                     # Weights original
    #                     W = (TLoc**2-dtmp_adaptive**2)**2/TLoc**4
    #                     W=W/np.sum(W,0) # normalization
    #                     Wfinal=np.tile(W, (radius.shape[0], 1)).T
    #                     wtmp_adaptive = Wfinal * tmp_adaptive

    #                     if ismom: # compute moments
    #                         Mom[0,index[igrid],:] = np.sum( tmp_adaptive**0       , axis=0)
    #                         Mom[1,index[igrid],:] = np.sum( wtmp_adaptive , axis=0)
    #                         Mom[2,index[igrid],:] = np.sum( wtmp_adaptive * tmp_adaptive, axis=0)
    #                         Mom[3,index[igrid],:] = np.sum( wtmp_adaptive * tmp_adaptive**2, axis=0)
    #                         Mom[4,index[igrid],:] = np.sum( wtmp_adaptive * tmp_adaptive**3, axis=0)
    #                         #print('grr')


    #                     if  len(dtmp_adaptive) >10: # compute ratio
    #                         # Non weighted std
    #                         NWstd[index[igrid]] = np.std( tmp_adaptive, axis=0, ddof=1)
    #                         # Non weighted dispersion index: 0.1 due to the edge problem
    #                         # weighted std
    #                         #Wstd[index[igrid]]  = np.std( wtmp_adaptive, axis=0, ddof=1)
    #                         #Wmean[index[igrid]] = np.mean( wtmp_fixed, axis=0)

    #                         average = np.average(tmp_adaptive, weights=Wfinal, axis=0)
    #                         Wmean[index[igrid]] = average
    #                         Wstd[index[igrid]]  = np.average((tmp_adaptive-average)**2, weights=Wfinal, axis=0)

    #                         # Non weighted ratio
    #                         stdtmp_adaptive=np.tile(NWstd[index[igrid]],(tmp_adaptive.shape[0], 1))
    #                         ttmp_adaptive = np.divide(tmp_adaptive, stdtmp_adaptive , out=np.zeros_like(tmp_adaptive), where=stdtmp_adaptive!=0)
    #                         ttmp_adaptive = np.sum(Wfinal * ttmp_adaptive,axis=0)
    #                         NWratio[index[igrid]] = ttmp_adaptive# np.sum( Wfinal * ttmp,axis=0)
    #                         # weighted ratio
    #                         stdtmp_adaptive = np.tile(Wstd[index[igrid]],(tmp_adaptive.shape[0], 1))
    #                         ttmp_adaptive = np.divide(tmp_adaptive , stdtmp_adaptive , out=np.zeros_like(tmp_adaptive), where=stdtmp_adaptive!=0)
    #                         ttmp_adaptive = np.sum(Wfinal * ttmp_adaptive,axis=0)
    #                         Wratio[index[igrid],:] = ttmp_adaptive #np.sum( Wfinal * ttmp,axis=0)
    #                         # weighted lacunarity
    #                         stdtmp_fixed = np.tile(Wstd[index[igrid]],(tmp_adaptive.shape[0], 1))
    #                         meantmp_fixed = np.tile(Wmean[index[igrid]],(tmp_adaptive.shape[0], 1))
    #                         ttmp_fixed = (np.divide(stdtmp_fixed , meantmp_fixed , out=np.zeros_like(tmp_adaptive), where=meantmp_fixed!=0))**2
    #                         ttmp_fixed = np.sum(Wfinal * ttmp_fixed,axis=0)
    #                         Wlacun[index[igrid],:] = ttmp_fixed #np.sum( Wfinal * ttmp,axis=0)

    # print(time.process_time() - start)
    # print('.')

    # print(np.allclose(Mom,Mom2))

    # print(np.allclose(NWstd,NWstd2))
    # print(np.allclose(Wmean,Wmean2))
    # print(np.allclose(Wstd,Wstd2))


    # print(np.allclose(Wratio,Wratio2))
    # print(np.allclose(NWratio,NWratio2))

    # print(np.allclose(Wlacun,Wlacun2))

    #%
    # cleaning ration (can be infinite if std==0)

    NWstd[~np.isfinite(NWstd)] = 0.
    Wstd[~np.isfinite(Wstd)] = 0.
    #NWratio[~np.isfinite(NWratio)] = 0.
    #Wratio[~np.isfinite(Wratio)] = 0.
    #Wlacun[~np.isfinite(Wlacun)] = 0.

    # pack output in list of arrays
    if ismom:
        out=[Wmean, Wstd, NWmean, NWstd, Mom]
        # out=[Wratio, Wstd, NWratio, NWstd, Mom, Wlacun]
    else:
        # out=[Wratio, Wstd, NWratio, NWstd, Wlacun]
        out=[Wmean, Wstd, NWmean, NWstd]

    return out
#%%
def WaveSmoothing(data,Count,X,Y,radius, T,  Nanalyse=2**16, k = 0, ismom=0, kernel = 0):
    """

   res = WaveSmoothing(data,Wave,X,Y,radius,T,Nanalyse=2**16, k = 0, ismom=0, kernel = 0))


   Compute kernel smoothing of the wavelet coefficient of a dataset of points.
   The geographical weighting is obtained using the bi-squared function of size T.
   The wavelet coeficients are obtained using the function GWMFA_count.m

   Input :

       data    - matrix of size Nx2 --> position (x,y) for N points
                 matrix of size Nx3 --> position (x,y,value) for N points
       Wave    - matrix of size Nxlength(radius) with wavelet count
                 oCan be obtained using the function  GWFA_count.m
       X       - array of dim 2 with x-postion of the grid nodes
       Y       - array of dim 2 with y-postion of the grid nodes : X and Y must have the same size
       radius  - list of scales to be investigated
       T       - bandwith of fixed kernel OR distance upper boundary of adaptive kernel
       k       - number of min neighbors for the adaptive kernel smoothing
       kernel  - if equals to 1 compute with adaptive kernel (default 0 which is fixed).
                 L_min = sqrt(2)*r_max, L_max = T, in between adaptive knn neighbors
      Nanalyse - number of points to analyse in one bach. Default is 2**16 points.
       isMom   - if equals to 1 compute also moment of order À to 4. Default 0.


   Output :

       res - list of two dimensional numpy matrix (of size length(data) X length(radius)

              res[Ø] = Wratio  : weighted ratio
              res[1] = Wstd    : weighted standert deviation
              res[2] = NWratio : non weighted ratio
              res[3] = NWstd   : non weighted stabdard deviation

            if isMom=1
              res[4] = Moo : weighted moment of order 0 to 4
                      matrix of size 5 X length(data) X length(radius)

              res[5] = WLacun  : weighted inverse ratio


    ##
     S.G.  Roux, ENS Lyon, December 2020,  stephane.roux@ens-lyon.fr
     J.L June 2021
    """


    si=data.shape
    if si[1] < 2 or si[1]>3:
        raise TypeError('The second dimension of first argument must be of length 2 or 3.')

    if Count.ndim != 2:
        raise TypeError('The second arument must be two dimensional.')

    if Count.shape[0] != si[0]:
        raise TypeError('The twofirst arument must have the same length.')

    if radius[radius>T].shape[0] >0:
        raise TypeError('The last argument must be greater than a sqrt(2)*radius.')

    # check grid
    if np.sum(np.abs(np.array(X.shape)-np.array(Y.shape)))>0:
        raise TypeError('X and Y must have the same size.')

    # create grid points
    mygrid=np.column_stack([X.flatten(),Y.flatten()])
    gridlength=mygrid.shape[0]
    gridsizecurr = X[0][1] - X[0][0]

    if Nanalyse==0:
        Nanalyse=gridlength
        Nbunch=1
    else:
        # find how many bunchs
        Nbunch=np.int(np.ceil(gridlength/Nanalyse))

    # results allocations
    if ismom:
        # mean wave
        Mom = np.zeros( (6, gridlength,radius.size), dtype=np.float)
        #Mom = np.zeros( (5, gridlength,radius.size), dtype=np.float)

    if k:
        k = k
    # NWstd wave
    NWstd = np.zeros( (gridlength,radius.size), dtype=np.float)
    # Wstd spatialized wave
    Wstd = np.zeros( (gridlength,radius.size), dtype=np.float)

    NWmean= np.zeros( (gridlength,radius.size), dtype=np.float)
    # Wstd spatialized wave
    Wmean = np.zeros( (gridlength,radius.size), dtype=np.float)

    # ratio wave
    NWratio = np.zeros( (gridlength,radius.size), dtype=np.float)
    Wratio  = np.zeros( (gridlength,radius.size), dtype=np.float)

    # inverse ratio wave
    NWlacun  = np.zeros( (gridlength,radius.size), dtype=np.float)
    Wlacun  = np.zeros( (gridlength,radius.size), dtype=np.float)


    # Loop on bunch
    for ibunch in range(Nbunch):
        print('bunch {:d}/{:d} '.format(ibunch+1,Nbunch), end=' ')

        # get data index of current bunch
        index=np.arange(ibunch*Nanalyse,(ibunch+1)*Nanalyse,1)
        index=index[index<gridlength]

        # we restrict the tree to points whithin a radius T (which must be > radius2)
        mi=np.min(mygrid[index,:], axis=0)
        ma=np.max(mygrid[index,:], axis=0)
        IndexNear=np.where((data[:,0] >mi[0]-T) & (data[:,0] <ma[0]+T) & (data[:,1]  >mi[1]-T) & (data[:,1] <ma[1]+T))[0]

        #if is_empty = False:
        tree = KDTree(data[IndexNear,0:2])

        # only compute value for grid points that contain raw data - eliminates artefacts at small T
        IdxMin = tree.query_radius(mygrid[index,:], r = np.sqrt(2)*gridsizecurr, count_only=True, return_distance=False)
        IdxMin = np.where(IdxMin>0.)


        IdxMin = np.unique(np.concatenate(IdxMin))
        index = index[IdxMin]

        if kernel == 0:
            neighbors_i_fixed, dist_ie_fixed = tree.query_radius(mygrid[index,:], r = T, count_only=False, return_distance=True, sort_results=False)


            print('.')
        else:
            neighbors_i_fixed, dist_ie_fixed = tree.query_radius(mygrid[index,:], r = T, count_only=False, return_distance=True, sort_results=False)
            dist_ie_adaptive, neighbors_i_adaptive = tree.query(mygrid[index,:], k = k, return_distance=True, sort_results=False)
            #print('.')

        for igrid in range (len(index)):

            # distance to current grid point
            dtmp_fixed = dist_ie_fixed[igrid]
            # coefficient of the grid point neighbors
            tmp_fixed = Count[IndexNear[neighbors_i_fixed[igrid]]]

            if kernel == 0:
                if len(dtmp_fixed) != 0:


                    # Weights original
                    W = (T**2-dtmp_fixed**2)**2/T**4
                    W=W/np.sum(W,0) # normalization

                    Wfinal=np.tile(W, (radius.shape[0], 1)).T
                    wtmp_fixed = Wfinal * tmp_fixed

                    if ismom: # compute moments
                        Mom[0,index[igrid],:] = np.sum( tmp_fixed**0       , axis=0)
                        Mom[1,index[igrid],:] = np.sum( np.abs(wtmp_fixed) , axis=0)
                        Mom[2,index[igrid],:] = np.sum( np.abs(wtmp_fixed) * np.abs(tmp_fixed), axis=0)
                        Mom[3,index[igrid],:] = np.sum( np.abs(wtmp_fixed) * tmp_fixed**2, axis=0)
                        Mom[4,index[igrid],:] = np.sum( np.abs(wtmp_fixed) * np.abs(tmp_fixed)**3, axis=0)
                        # Flatness
                        Mom[5,index[igrid],:] = np.sum(np.abs(tmp_fixed)**4, axis=0)/np.sum(np.abs(tmp_fixed)**2, axis=0)/3
                        #print('OK')

                    if  len(dtmp_fixed) >10: # compute ratio
                        # Non weighted std
                        NWstd[index[igrid]] = np.std( tmp_fixed, axis=0, ddof=1)
                        NWmean[index[igrid]] = np.mean( tmp_fixed, axis=0)
                        # Non weighted dispersion index: 0.1 due to the edge problem

                        # weighted std
                        Wstd[index[igrid]]  = np.std( wtmp_fixed, axis=0, ddof=1)
                        Wmean[index[igrid]] = np.mean( wtmp_fixed, axis=0)

                        # Non weighted ratio
                        stdtmp_fixed=np.tile(NWstd[index[igrid]],(tmp_fixed.shape[0], 1))
                        ttmp_fixed = np.divide(tmp_fixed, stdtmp_fixed , out=np.zeros_like(tmp_fixed), where=stdtmp_fixed!=0)
                        ttmp_fixed = np.sum(Wfinal * ttmp_fixed,axis=0)
                        NWratio[index[igrid]] = ttmp_fixed

                        # weighted ratio
                        stdtmp_fixed = np.tile(Wstd[index[igrid]],(tmp_fixed.shape[0], 1))
                        ttmp_fixed = np.divide(tmp_fixed , stdtmp_fixed , out=np.zeros_like(tmp_fixed), where=stdtmp_fixed!=0)
                        ttmp_fixed = np.sum(Wfinal * ttmp_fixed,axis=0)
                        Wratio[index[igrid],:] = ttmp_fixed

                        # weighted lacunarity
                        stdtmp_fixed = np.tile(Wstd[index[igrid]],(tmp_fixed.shape[0], 1))
                        meantmp_fixed = np.tile(Wmean[index[igrid]],(tmp_fixed.shape[0], 1))
                        ttmp_fixed = (np.divide(stdtmp_fixed , meantmp_fixed , out=np.zeros_like(tmp_fixed), where=meantmp_fixed!=0))**2
                        ttmp_fixed = np.sum(Wfinal * ttmp_fixed,axis=0)
                        Wlacun[index[igrid],:] = ttmp_fixed

            else:
                if len(dtmp_fixed != 0):
                    dtmp_adaptive= dist_ie_adaptive[igrid]
                    # coefficient of the grid point neighbors
                    tmp_adaptive = Count[IndexNear[neighbors_i_adaptive[igrid]]]

                    # Weights =
                    TLoc = np.max(dtmp_adaptive)
                    if TLoc < np.sqrt(2)*radius.max():
                        #break
                        raise TypeError("Local environment is smaller than radius max --> choose larger k")
                    elif TLoc > T:

                        # Weights original
                        W = (T**2-dtmp_fixed**2)**2/T**4
                        W=W/np.sum(W,0) # normalization

                        Wfinal=np.tile(W, (radius.shape[0], 1)).T
                        wtmp_fixed = Wfinal * tmp_fixed


                        if ismom: # compute moments
                            Mom[0,index[igrid],:] = np.sum( tmp_fixed**0       , axis=0)
                            Mom[1,index[igrid],:] = np.sum( np.abs(wtmp_fixed) , axis=0)
                            Mom[2,index[igrid],:] = np.sum( np.abs(wtmp_fixed) * np.abs(tmp_fixed), axis=0)
                            Mom[3,index[igrid],:] = np.sum( np.abs(wtmp_fixed) * tmp_fixed**2, axis=0)
                            Mom[4,index[igrid],:] = np.sum( np.abs(wtmp_fixed) * np.abs(tmp_fixed)**3, axis=0)



                        if  len(dtmp_fixed) >10: # compute ratio
                            # Non weighted std
                            NWstd[index[igrid]] = np.std( tmp_fixed, axis=0, ddof=1)
                            # Non weighted dispersion index: 0.1 due to the edge problem
                            # weighted std
                            Wstd[index[igrid]]  = np.std( wtmp_fixed, axis=0, ddof=1)
                            Wmean[index[igrid]] = np.mean( wtmp_fixed, axis=0)
                            # Non weighted ratio
                            stdtmp_fixed=np.tile(NWstd[index[igrid]],(tmp_fixed.shape[0], 1))
                            ttmp_fixed = np.divide(tmp_fixed, stdtmp_fixed , out=np.zeros_like(tmp_fixed), where=stdtmp_fixed!=0)
                            ttmp_fixed = np.sum(Wfinal * ttmp_fixed,axis=0)
                            NWratio[index[igrid]] = ttmp_fixed # np.sum( Wfinal * ttmp,axis=0)
                            # weighted ratio
                            stdtmp_fixed = np.tile(Wstd[index[igrid]],(tmp_fixed.shape[0], 1))
                            ttmp_fixed = np.divide(tmp_fixed , stdtmp_fixed , out=np.zeros_like(tmp_fixed), where=stdtmp_fixed!=0)
                            ttmp_fixed = np.sum(Wfinal * ttmp_fixed,axis=0)
                            Wratio[index[igrid],:] = ttmp_fixed #np.sum( Wfinal * ttmp,axis=0)
                            # weighted lacunarity
                            stdtmp_fixed = np.tile(Wstd[index[igrid]],(tmp_fixed.shape[0], 1))
                            meantmp_fixed = np.tile(Wmean[index[igrid]],(tmp_fixed.shape[0], 1))
                            ttmp_fixed = (np.divide(stdtmp_fixed , meantmp_fixed , out=np.zeros_like(tmp_fixed), where=meantmp_fixed!=0))**2
                            ttmp_fixed = np.sum(Wfinal * ttmp_fixed,axis=0)
                            Wlacun[index[igrid],:] = ttmp_fixed #np.sum( Wfinal * ttmp,axis=0)

                    else:

                        # Weights original
                        W = (TLoc**2-dtmp_adaptive**2)**2/TLoc**4
                        W=W/np.sum(W,0) # normalization
                        Wfinal=np.tile(W, (radius.shape[0], 1)).T
                        wtmp_adaptive = Wfinal * tmp_adaptive

                        if ismom: # compute moments
                            Mom[0,index[igrid],:] = np.sum( tmp_adaptive**0       , axis=0)
                            Mom[1,index[igrid],:] = np.sum( np.abs(wtmp_adaptive) , axis=0)
                            Mom[2,index[igrid],:] = np.sum( np.abs(wtmp_adaptive) * np.abs(tmp_adaptive), axis=0)
                            Mom[3,index[igrid],:] = np.sum( np.abs(wtmp_adaptive) * tmp_adaptive**2, axis=0)
                            Mom[4,index[igrid],:] = np.sum( np.abs(wtmp_adaptive) * np.abs(tmp_adaptive)**3, axis=0)



                        if  len(dtmp_adaptive) >10: # compute ratio
                            # Non weighted std
                            NWstd[index[igrid]] = np.std( tmp_adaptive, axis=0, ddof=1)
                            # Non weighted dispersion index: 0.1 due to the edge problem
                            # weighted std
                            Wstd[index[igrid]]  = np.std( wtmp_adaptive, axis=0, ddof=1)
                            Wmean[index[igrid]] = np.mean( wtmp_fixed, axis=0)
                            # Non weighted ratio
                            stdtmp_adaptive=np.tile(NWstd[index[igrid]],(tmp_adaptive.shape[0], 1))
                            ttmp_adaptive = np.divide(tmp_adaptive, stdtmp_adaptive , out=np.zeros_like(tmp_adaptive), where=stdtmp_adaptive!=0)
                            ttmp_adaptive = np.sum(Wfinal * ttmp_adaptive,axis=0)
                            NWratio[index[igrid]] = ttmp_adaptive# np.sum( Wfinal * ttmp,axis=0)
                            # weighted ratio
                            stdtmp_adaptive = np.tile(Wstd[index[igrid]],(tmp_adaptive.shape[0], 1))
                            ttmp_adaptive = np.divide(tmp_adaptive , stdtmp_adaptive , out=np.zeros_like(tmp_adaptive), where=stdtmp_adaptive!=0)
                            ttmp_adaptive = np.sum(Wfinal * ttmp_adaptive,axis=0)
                            Wratio[index[igrid],:] = ttmp_adaptive #np.sum( Wfinal * ttmp,axis=0)
                            # weighted lacunarity
                            stdtmp_fixed = np.tile(Wstd[index[igrid]],(tmp_fixed.shape[0], 1))
                            meantmp_fixed = np.tile(Wmean[index[igrid]],(tmp_fixed.shape[0], 1))
                            ttmp_fixed = (np.divide(stdtmp_fixed , meantmp_fixed , out=np.zeros_like(tmp_fixed), where=meantmp_fixed!=0))**2
                            ttmp_fixed = np.sum(Wfinal * ttmp_fixed,axis=0)
                            Wlacun[index[igrid],:] = ttmp_fixed #np.sum( Wfinal * ttmp,axis=0)


    print('.')

    # cleaning ration (can be infinite if std==0)
    NWratio[~np.isfinite(NWratio)] = 0.
    Wratio[~np.isfinite(Wratio)] = 0.
    NWstd[~np.isfinite(NWstd)] = 0.
    Wstd[~np.isfinite(Wstd)] = 0.
    Wlacun[~np.isfinite(Wlacun)] = 0.

    # pack output in list of arrays
    if ismom:
        out=[Wratio, Wstd, NWratio, NWstd, Mom, Wlacun]
    else:
        out=[Wratio, Wstd, NWratio, NWstd, Wlacun]


    return out
# %%
def analysis(data,radius,gridsize,multT,center=0,sizetot=[]):
    """

       ##
     S.G.  Roux, ENS Lyon, December 2020,  stephane.roux@ens-lyon.fr
    """
    # check input argument
    if data.ndim != 2:
        raise TypeError('The first arument must be two dimensional.')

    si=data.shape
    if si[1] < 2 or si[1]>3:
        raise TypeError('The second dimension of first argument must be of length 2 or 3.')

    if not isinstance(gridsize, int) or gridsize<0:
        raise TypeError('The third argument  must be a positive integer.')

    if multT<2:
        raise TypeError('The fourth argument must be larger than 2.')

    radius=np.array(radius)
    if np.sum(radius>0)-radius.shape[0] != 0:
        raise TypeError('The fifth argument must be a vector of positive nember.')

    T=gridsize*multT
    if np.sum(np.sqrt(2)*radius>T)>.5:
        raise TypeError('gridsize*multT must be greater than sqrt(2)*radius.')

    # centering the data in a square of size 2*sizetot
    center=np.array(center)
    sizetot=np.array(sizetot)
    if center.ndim==2: # we nedd to
        if len(sizetot)==0:
            raise TypeError('center and sizetot must be given together.')
        points=data[np.abs(data[:,0] - center[0]<sizetot+T) & (np.abs(data[:,1]) - center[1]<sizetot+T)]
        data=points

        print('Centering : we kept {:d} points on {:d} ({:.2f}).'.format(data.shape[0],si[0],data.shape[0]/si[0]*100))
        if data.shape[0] < 1000 :
            raise TypeError('The number of data point seem quiet small (below 1000).')

    # create grid points
    print('Creating grid.')
    x = np.arange(data[:,0].min(),data[:,0].max(),gridsize)
    y = np.arange(data[:,1].min(),data[:,1].max(),gridsize)
    # 2 dimensional grid
    X,Y = np.meshgrid(x,y)
    print('The grid size will be {:d} by {:d}.'.format(X.shape[0],X.shape[1]))

    # wavelet coefficients computation
    print('First : wavelet coefficient computation.')
    WT, BC = localWaveTrans(data, radius)

    # geographical smoothing
    print('Second : geographical weighting.')
    out = WaveSmoothing(data,WT,X,Y,radius,T)

    return out

# %%
if __name__ == '__main__':
    # %% testing area

    # parameters
    # size od the window analysis
    sizetot=25000
    # size of the grid en metre
    gridsize=200
    T=4*gridsize
    # radius choice
    radius=np.array([2, 4, 8, 16, 32, 64, 128, 256, 512])

    # centering data
    ## find coordinate at https://epsg.io
    # select center of analysis : Paris
    # center_epsg3035=np.array([3760363.478534,   2889272.871095]) # ETRS89 LAEA
    # center_epsg2154=np.array([652082.05, 6861785.42])  # RGF93
    # center=center_epsg2154

    # select center of analysis : Lyon
    # center_epsg3035=np.array([3922020.710,  2528599.291])  # ETRS89 LAEA
    center_epsg2154= np.array([ 845783.55788093, 6518049.99703196]) # RGF93-lamber93
    center=center_epsg2154

    # select center of analysis : Marseille
    # center_epsg3035=np.array([3945391.555, 2253009.607])
    # center_epsg2154= np.array([893414.80662366, 6245355.01525851])
    # center=center_epsg2154

    # %% read and select data
    tmp = np.genfromtxt('/Users/sroux/Disque2/CURRENT/Population/Batiment/batiments.csv', delimiter=',')
    # transform to geodataframe
    df = pd.DataFrame.from_records(tmp)
    df=df.rename(columns={0:"x", 1:"y"})
    # delete temp variable
    del tmp
    # get the image under analyze
    points=df.loc[(np.abs(df.x - center[0])<sizetot) & (np.abs(df.y - center[1])<sizetot)]
    del df

    data=points.to_numpy()
    # count points in radius or sum values store in points.loc[:,2]
    WT, BC = localWaveTrans(data, radius)
    uu=np.random.randn(409952, 3)
    uu[:,0:2]=data
    WT2, BC2 = localWaveTrans(uu, radius)

    # create the grid
    x = np.arange(data[:,0].min(),data[:,0].max(),gridsize)
    y = np.arange(data[:,1].min(),data[:,1].max(),gridsize)
    X,Y = np.meshgrid(x,y)
    mygrid=np.column_stack([X.flatten(),Y.flatten()])
    gridlength=len(x)*len(y)

    # wave smoothing
    out = WaveSmoothing(data,WT[:,0:8],X,Y,radius[0:8],T)
    out2 = WaveSmoothing(uu,WT2[:,0:8],X,Y,radius[0:8],T)

    # display numpy array
    Wratio = out[0]
    uu = Wratio.reshape((250, 250, 8))
    plt.imshow(uu[:,:,6])
    plt.colorbar()

    # convert to panda frame
    gdf_Wratio  = restoGeoPandaFrame(mygrid, radius[0:8] ,out[0])
    gdf_Wstd    = restoGeoPandaFrame(mygrid, radius[0:8] ,out[1])
    gdf_NWratio = restoGeoPandaFrame(mygrid, radius[0:8] ,out[2])
    gdf_NWstd   = restoGeoPandaFrame(mygrid, radius[0:8] ,out[3])

    gdf_Wratio2  = restoGeoPandaFrame(mygrid, radius[0:8] ,out2[0])
    # display geodataframe
    fig, ax = plt.subplots(figsize=(20,25))
    #gdf_grid.loc[gdf_grid.std_r100 != 0.0].plot(column='std_r100',  scheme='naturalbreaks', k=10, ax=ax, legend=True, cmap = "viridis_r", edgecolor = 'face', linewidth=0.0001)
    gdf_Wratio.loc[gdf_Wratio.R4 != 0.0].plot(column='R4',  k=10, ax=ax, legend=True, cmap = "viridis_r", edgecolor = 'face', linewidth=0.0001)

    fig, ax = plt.subplots(figsize=(20,25))
    gdf_Wratio.plot(column='R64',  k=10, ax=ax, legend=True, cmap = "viridis_r", edgecolor = 'face', linewidth=0.0001)

    fig, ax = plt.subplots(figsize=(20,25))
    gdf_Wratio2.plot(column='R64',  k=10, ax=ax, legend=True, cmap = "viridis_r", edgecolor = 'face', linewidth=0.0001)

    # if moment computation : ismon = 1
    # if ismon == 1
    #     MomW=out[4] and MomC=out[5]
    Wratio=out[0]
    Wstd=out[1]
    NWratioC=out[2]
    NWstdC=out[3]
    # if ismon == 1 then MomW=out[4] and MomC=out[5]


    # %% full analysis in one shot
    # multT=2 # define T=mulT*gridsize
    # out=analysis(data,radius[0:8],gridsize,multT)
