# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 17:59:21 2015

@author: areed145
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

file = '/Users/areed145/Documents/Python/temp_sat_tool/pickles/CYMRICMCKITTRICK_06_all_log_compiled.pk1'
mlc = pd.read_pickle(file)
    
xres = 250
yres = 250
zres  = 150
    
mlc['TVDSS_U'] = np.round((mlc['TVDSS'] / zres),0) * zres
mlc['X_U'] = np.round((mlc['X'] / xres),0) * xres
mlc['Y_U'] = np.round((mlc['Y'] / yres),0) * yres

gb = mlc.groupby(['TVDSS_U','X_U','Y_U'])

df_res = gb.RES.prod() ** (1 / gb.RES.count())
df_res = df_res.reset_index()


df_temp = gb.TEMP_NEW.prod() ** (1 / gb.TEMP_NEW.count())
df_temp = df_temp.reset_index()

df_nphi = gb.NPHI_NEW.prod() ** (1 / gb.NPHI_NEW.count())
df_nphi = df_nphi.reset_index()

df_all = pd.merge(df_res, df_temp, how='outer', on=['TVDSS_U','X_U','Y_U'])
df_all = pd.merge(df_all, df_nphi, how='outer', on=['TVDSS_U','X_U','Y_U'])
df_all = df_all.replace([np.inf, -np.inf], np.nan).dropna(subset=['RES','TEMP_NEW','NPHI_NEW'], how="all")

df_res_modl = df_all[pd.notnull(df_all['RES'])]
df_res_pred = df_all[pd.isnull(df_all['RES'])]

rbfi = Rbf(df_res_modl.X_U, df_res_modl.Y_U, df_res_modl.TVDSS_U, df_res_modl.RES)  # radial basis function interpolator instance

df_res_pred.RES = rbfi(df_res_pred.X_U, df_res_pred.Y_U, df_res_pred.TVDSS_U)   # interpolated values

df_res_ = df_res_modl.append(df_res_pred)

plt.figure(figsize=(20,20))
plt.title('PROPERTY PLOT')

plt.subplot(2,2,1)
plt.scatter(df_all.X_U, df_all.Y_U, s=(xres+yres+zres)/4, alpha=.8, c=df_all.TEMP_NEW, cmap='jet', lw = 0)
plt.grid(b=True, which='major', color='b', linestyle='-')
plt.grid(b=True, which='minor', color='r', linestyle='--')    
plt.xlabel('X') 
plt.ylabel('Y')    
plt.xlim(1504000,1511000)
plt.ylim(673000,679000)

plt.subplot(2,2,3)
plt.scatter(df_all.X_U, df_all.TVDSS_U, s=(xres+yres+zres)/4, alpha=.8, c=df_all.TEMP_NEW, cmap='jet', lw = 0)
#plt.xscale('log')
plt.grid(b=True, which='major', color='b', linestyle='-')
plt.grid(b=True, which='minor', color='r', linestyle='--')    
plt.xlabel('X') 
plt.ylabel('TVDSS')    
plt.xlim(1504000,1511000)
plt.ylim(-2000,1000)

plt.subplot(2,2,2)
plt.scatter(df_all.TVDSS_U, df_all.Y_U, s=(xres+yres+zres)/4, alpha=.8, c=df_all.TEMP_NEW, cmap='jet', lw = 0)
#plt.xscale('log')
plt.grid(b=True, which='major', color='b', linestyle='-')
plt.grid(b=True, which='minor', color='r', linestyle='--')    
plt.xlabel('TVDSS') 
plt.ylabel('Y')    
plt.xlim(-2000,1000)
plt.ylim(673000,679000)

plt.savefig('scatter.png',dpi=480)
plt.show()