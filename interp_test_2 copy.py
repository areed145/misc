# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 21:34:49 2015

@author: areed145
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.mlab import griddata
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

sand = 'AMN'
prop = 'TOP_TVDSS'
dx, dy = 0.05, 0.05

df = pd.read_pickle('/Users/areed145/Documents/Python/temp_sat_tool/pickles/CYMRICMCKITTRICK_all_log_summary.pk1')
df = df[df.SAND == sand]
df = df[pd.notnull(df[prop])]
df = df[['TOP_X', 'TOP_Y', prop]]

x = df['TOP_X'].values.T
y = df['TOP_Y'].values.T
z = df[prop].values.T

# make these smaller to increase the resolution
dx, dy = 100, 100

# generate 2 2d grids for the x & y bounds
yi, xi = np.mgrid[np.linspace(df.TOP_Y.min(), df.TOP_Y.max()+dy, dy),
                np.linspace(df.TOP_X.min(), df.TOP_X.max()+dx, dx)]
                
xx = np.linspace(df.TOP_X.min(), df.TOP_X.max()+dx, dx)
yy = np.linspace(df.TOP_Y.min(), df.TOP_Y.max()+dy, dy)                

zi = griddata(x, y, z, xx, yy, interp='linear')

# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
zi = zi[:-1, :-1]
levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())

# pick the desired colormap, sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
cmap = plt.get_cmap('PiYG')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

plt.subplot(2, 1, 1)
im = plt.pcolormesh(xi, yi, zi, cmap=cmap, norm=norm)
plt.colorbar()
# set the limits of the plot to the limits of the data
plt.axis([xi.min(), xi.max(), yi.min(), yi.max()])
plt.title('pcolormesh with levels')

plt.subplot(2, 1, 2)
# contours are *point* based plots, so convert our bound into point
# centers
plt.contourf(xi[:-1, :-1] + dx / 2.,
             yi[:-1, :-1] + dy / 2., zi, levels=levels,
             cmap=cmap)
plt.colorbar()
plt.title('contourf with levels')

plt.show()