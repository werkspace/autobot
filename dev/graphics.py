#!/usr/bin/env python
# coding : utf-8

import numpy as np
import matplotlib.pyplot as plt

import geometry as geo

import pdb

def plot_frame(origin, basis, ax=None, *args, **kwargs):
    """
    Draw an arrow representing an orthogonal frame defined by its origin and basis.

        :param origin: a (2,1) vector representing the origin of the frame to be drawn.
        :param basis: a (2,2) vector representing the basis of the frame to be drawn.

        :type origin: a numpy nd-array.
        :type basis: a numpy nd-array.

    """
    # Compute the image of the identity and the origin in the target frame.
    # The former will be the origin of the arrow, the latter the end of the arrow.
    frame_origin = geo.to_frame(origin, basis, np.zeros((2)))
    frame_identity = geo.to_frame(origin, basis, np.array([1,1]))

    # Getting x and y columns
    x, y = frame_origin.T
    dx, dy = frame_identity.T - frame_origin.T  
    
    # This draws an arrow from (x, y) to (x+dx, y+dy)
    if ax is None:
        plt.quiver(x, y, dx, dy, scale_units='xy', angles='xy', scale=1)
    else:
        ax.quiver(x, y, dx, dy, scale_units='xy', angles='xy', scale=1) 

if __name__ == '__main__':

    title = 'Graphics library demo'
    print(title)

    origin = np.array([5, 5])
    basis = np.array([[-1, 0], [0, -1]])

    fig, ax = plt.subplots()
    ax.plot(0, 0, 'k+')
    ax.set_title(title)

    plot_frame(origin=origin, basis=basis, ax=ax)
    
    plt.show()
