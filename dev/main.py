#!/usr/bin/env python
# coding : utf-8
#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import dataset
from geometry import cartesian_to_polar
from model import ransac_line_extraction

#%% Setup
x_sensor, y_sensor = dataset.x_sensor, dataset.y_sensor
x_world, y_world = dataset.x_world, dataset.y_world

total_readings = np.stack([x_sensor, y_sensor])

# Centering on (200,200), and converting to polar
total_readings = total_readings - 200
r_sensor, theta_sensor = cartesian_to_polar(total_readings[0,:], total_readings[1,:])
total_readings = np.vstack((total_readings, r_sensor, theta_sensor))

# Algorithm hyperparameter
n_iter = 2000
k_neighbours = 15
degrees_range = (15.0 / 360.0) * (2 * np.pi )
consensus = k_neighbours * .7
tolerance = 100 # squared meter

print('World size :', x_world, y_world)
print('Number of readings :', len(total_readings[1]))
print('Number of neighbours :', k_neighbours)
print('Consensus :', consensus)


#%% Line extraction

line_models = ransac_line_extraction(data=total_readings,
                                    k_neighbours=k_neighbours,
                                    degrees_range=degrees_range,
                                    consensus=consensus,
                                    tolerance=tolerance,
                                    n_iter=n_iter)

print('%d lines found' % len(line_models))

plt.figure(figsize=(20,10))
plt.plot(total_readings[0], total_readings[1], 'ko')

n = len(line_models)
color = iter(cm.rainbow(np.linspace(0,1,n)))

for m in line_models:
    c = next(color)
    x_min = m.fitted[0].min()
    x_max = m.fitted[0].max()
    x = np.linspace(x_min, x_max)
    plt.plot(x, m.slope * x + m.interceipt, c=c)
    plt.plot(m.points[0], m.points[1], '*', c=c)

plt.xlim(-200, 200)
plt.ylim(-200, 300)

plt.show()
