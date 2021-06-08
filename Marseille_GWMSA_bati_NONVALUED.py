#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 10:59:20 2021

@author: sroux
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 20:24:10 2021

@author: sroux
"""

#!/usr/bin/env python
# coding: utf-8

# In[1]:
#%% using sklearn search instead --> multithreading --> NOT INTERESTING
# import numpy as np
# from sklearn.neighbors import NearestNeighbors

# samples = [[0., 0.], [0., .5], [1., 1.]]
# neigh = NearestNeighbors(radius=1.6,n_jobs=2)
# neigh.fit(samples)
# #NearestNeighbors(radius=1.6)
# dist, index = neigh.radius_neighbors([[1., 1.]],return_distance=True)
# # %%
# import time

# start = time.process_time()
# neigh = NearestNeighbors(radius=1.6)
# neigh.fit(data)
# dist, index = neigh.radius_neighbors(data[1:2**16,:],radius=100,return_distance=True)
# print(time.process_time() - start)


# start = time.process_time()
# treetot = KDTree(data)
# Idxtot , Disttot = treetot.query_radius(data[1:2**16,0:2], r = 100,count_only=False,return_distance=True)           
# print(time.process_time() - start)
      
# %%
import sys
sys.stdout.flush()

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import pandas as pd
import geopandas as gpd

from numpy.polynomial import polynomial as P



# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


#in main folder
import GWMSA_June2021_optimized as GWMSA




# fp1='/Users/sroux/Disque2/CURRENT/Geographie/Projet_MF_Upload/2_Data/5_Boundaries/lyon_metro/lyon_metro.shp'
# #p1 = '/Users/jankalengyel/2_ENS/3_Data/6_Boundaries/adr_voie_lieu.shp/lyon_metro.shp'
# outline_lyon = gpd.read_file(fp1)
# outline_lyon = outline_lyon.to_crs({'init': 'epsg:2154'})
# bounding_box_outline_lyon = outline_lyon.envelope
# df_bounding_box_outline_lyon = gpd.GeoDataFrame(gpd.GeoSeries(bounding_box_outline_lyon), columns=['geometry'], crs={'init': 'epsg:2154'})
# center=np.array([bounding_box_outline_lyon.centroid.x,bounding_box_outline_lyon.centroid.y])




# fp1 = '/Users/jankalengyel/2_ENS/3_Data/6_Boundaries/adr_voie_lieu.shp/lyon_city.shp'
# city_lyon = gpd.read_file(fp1)
# city_lyon = city_lyon.to_crs({'init': 'epsg:2154'})


#  read data and creta data frame

df = pd.read_csv('../3cities_bati/Marseille_50km_bati_May21.csv')

gdf_50k_bati = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.x, df.y))


gdf_50k_bati.crs = 'epsg:2154'
gdf_50k_bati.columns

gdf_50k_bati.HAUTEUR.fillna(0, inplace = True)

# # Prepare the grid, define radiuses

gridsize=50

x = np.arange(gdf_50k_bati['x'].min()+gridsize/2,gdf_50k_bati['x'].max()-gridsize/2,gridsize)
y = np.arange(gdf_50k_bati['y'].min()+gridsize/2,gdf_50k_bati['y'].max()-gridsize/2,gridsize)

X,Y = np.meshgrid(x,y)
mygrid=np.column_stack([X.flatten(),Y.flatten()])

pf_my_grid=pd.DataFrame.from_records(mygrid)
pf_my_grid=pf_my_grid.rename(columns={0:"x", 1:"y"})
    

# parameters radius 
radius = np.array([ 50., 75., 100., 125., 150., 175., 200., 225., 250., 275., 300., 325., 350., 375., 400.])#650

T=13.*gridsize

data=np.column_stack([gdf_50k_bati.x, gdf_50k_bati.y])


# %% OPTIMIZED

radius = np.array([ 50., 75., 100., 125., 150., 175., 200., 225., 250., 275., 300., 325., 350., 375., 400.])#650

T=13.*gridsize

# Normalized coef
WT, BC, BC_T = GWMSA.localWaveTrans(data, radius, T, glob = 1)
WT_Norm=WT/np.tile(BC_T,(len(radius),1)).transpose()
BC_Norm=BC/np.tile(BC_T,(len(radius),1)).transpose()

# out=[Wratio, Wstd, NWratio, NWstd, Mom, Wlacun]
out = GWMSA.WaveSmoothingOptim(data,WT,X,Y,radius, T, ismom=1, kernel=0, Nanalyse=2**15)
out_Norm = GWMSA.WaveSmoothingOptim(data,WT_Norm,X,Y,radius, T, ismom=1, kernel=0, Nanalyse=2**15)

np.savez('Marseille_optim_resuls',WT=WT,BC=BC, BC_T=BC_T,out=out,out_Norm=out_Norm,radius=radius,T=T)



# %% get the data


radius=np.linspace(50,1000,20,endpoint=True)
T=2000
# from optimize --> one more output : CountGlobalon T
#WT, BC = GWMSA.localWaveTrans(data, radius, T) 


#np.savez('results_largeR_T2000_Marseille', WT=WT, BC=BC,radius=radius,T=T)

# %%
npzfile = np.load('results_largeR_T2000_Marseille.npz')
WT = npzfile['WT']
BC = npzfile['BC']
radius = npzfile['radius']
T = npzfile['T']

# %% Moment
M2, Fl, Sk, Mean, MeanAbs = computeM2FlatnessSkewness(WT,radius)
M2_BC, Fl_BC, Sk_BC, Mean_BC, MeanAbs_BC = computeM2FlatnessSkewness(BC,radius)


fig, ax=plt.subplots(3,2) 
uuM2=np.polyfit(np.log(radius[3:12]),np.log(M2[3:12]),1)
ax[0,0].plot(np.log(radius),np.log(M2),'ko',label='Not Normalized coef {:.2f}'.format(uuM2[0]))
ax[0,0].plot(np.log(radius),np.polyval(uuM2,np.log(radius)),'r')

ax[0,0].set_title('M2 WT ')
ax[0,0].legend()

uuM2_BC=np.polyfit(np.log(radius[:5]),np.log(M2_BC[:5]),1)
ax[0,1].plot(np.log(radius),np.log(M2_BC),'ko',label='Not Normalized BC  {:.2f}'.format(uuM2_BC[0]))
ax[0,1].plot(np.log(radius),np.polyval(uuM2_BC,np.log(radius)),'r')

ax[0,1].legend()
ax[0,1].set_title('M2 BC')

uuF=np.polyfit(np.log(radius[3:12]),np.log(Fl[3:12]),1)
ax[1,0].plot(np.log(radius),np.log(Fl),'ko',label='Not Normalized coef {:.2f}'.format(uuF[0]))
ax[1,0].plot(np.log(radius),np.polyval(uuF,np.log(radius)),'r')

ax[1,0].set_title('Flatness WT')
ax[1,0].legend()

uuF_BC=np.polyfit(np.log(radius[:5]),np.log(Fl_BC[:5]),1)
ax[1,1].plot(np.log(radius),np.log(Fl_BC),'ko',label='Not Normalized BC {:.2f}'.format(uuF_BC[0]))
ax[1,1].plot(np.log(radius),np.polyval(uuF_BC,np.log(radius)),'r')

uud=np.polyfit(np.log(radius[3:12]),np.log(MeanAbs[3:12]),1)
ax[2,0].plot(np.log(radius),np.log(MeanAbs),'ko',label='Not Normalized coef {:.2f}'.format(uud[0]))
ax[2,0].plot(np.log(radius),np.polyval(uud,np.log(radius)),'r')

ax[2,0].set_title('M1 WT')
ax[2,0].legend()

uud_BC=np.polyfit(np.log(radius[:5]),np.log(MeanAbs_BC[:5]),1)
ax[2,1].plot(np.log(radius),np.log(MeanAbs_BC),'ko',label='Not Normalized BC {:.2f}'.format(uud_BC[0]))
ax[2,1].plot(np.log(radius),np.polyval(uud_BC,np.log(radius)),'r')

ax[2,1].legend()
ax[2,1].set_title('M1 BC')
plt.tight_layout()
#plt.savefig('myFlatness.png')
fig.suptitle('Marseille')
plt.savefig('Marseille_WT_MC_flatness.png')


print('Not normalized WT')
print('Estimated H  {:.2f}'.format(uuM2[0]/2))
print('Estimated lambda2  {:.2f}'.format(-1*uuF[0]/4))
print('Not normalized BC')
print('Estimated H  {:.2f}'.format(uuM2_BC[0]/2))
print('Estimated lambda2  {:.2f}'.format(-1*uuF_BC[0]/4))


# %% histograms
nbins=128
hist, centers, lastd = computehistogram(WT,nbins)
hist_BC, centers_BC, lastd_BC = computehistogram(BC,nbins)


dec=0
fig, ax=plt.subplots(2,1) 
for ir in range(5,len(radius)):
    ax[0].plot(centers[:,ir],np.log(hist[:,ir])-ir*dec)
    ax[0].set_title('Marseille : coef')
    ax[1].plot(centers[:,ir],np.log(hist_BC[:,ir])-ir*dec)
    ax[1].set_title('Marseille :  BC')
plt.savefig('Hist_Marseille.png')



