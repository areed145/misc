# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 21:34:49 2015

@author: areed145
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.mlab import griddata

sand = 'AMN'
prop = 'MIN_TEMP'
dx, dy = 0.05, 0.05

df = pd.read_pickle('/Users/areed145/Documents/Python/temp_sat_tool/pickles/CYMRICMCKITTRICK_all_log_summary.pk1')
df = df[df.SAND == sand]
df = df[pd.notnull(df[prop])]
df = df[['TOP_X', 'TOP_Y', prop]]

x = df['TOP_X'].values.T
y = df['TOP_Y'].values.T
z = df[prop].values.T
# define grid.
xi = np.linspace(df.TOP_X.min(), df.TOP_X.max(), 100)
yi = np.linspace(df.TOP_Y.min(), df.TOP_Y.max(), 100)
# grid the data.
zi = griddata(x, y, z, xi, yi, interp='nn')
# contour the gridded data, plotting dots at the nonuniform data points.
CS = plt.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
CS = plt.contourf(xi, yi, zi, 15, cmap=plt.cm.rainbow,
                  vmax=zi.max(), vmin=zi.min())
plt.colorbar()  # draw colorbar
# plot data points.
plt.scatter(x, y, marker='o', c='b', s=5, zorder=10)
plt.title(sand+': '+prop)
plt.show()