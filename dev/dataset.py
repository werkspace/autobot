#!/usr/bin/env python
# coding : utf-8

import numpy as np
import matplotlib.pyplot as plt

world = plt.imread('src/simple_map.png')

world = world[:,:,0]
world = 1 - world

# Convert gridmap matrix into cartesian coordinates
x, y = np.where(world > 0)

n = x.shape[0]
n_sample = 500

x_world, y_world = world.shape

sample_idx = np.random.randint(0, n, size=n_sample)
x_err, y_err = np.random.standard_cauchy(size=(2,n_sample))

x_sensor = (x[sample_idx] + x_err).clip(50, x_world)
y_sensor = (y[sample_idx] + y_err).clip(0, y_world)