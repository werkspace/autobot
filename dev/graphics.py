#!/usr/bin/env python
# coding : utf-8

import numpy as np
import matplotlib.pyplot as plt

import geometry as geo

import pdb

def plot_frame(origin, theta, ax=None, *args, **kwargs):
    """
    Draw an arrow representing an orthogonal frame defined by its origin and basis.

        :param origin: a (2,1) vector representing the origin of the frame to be drawn.
        :param theta: an orientation angle.

        :type origin: a numpy nd-array.
        :type theta: a float.

    """
    # Compute the image of the identity and the origin in the target frame.
    # The former will be the origin of the arrow, the latter the end of the arrow.
    frame_origin = geo.to_frame(origin, theta, np.zeros((2)))
    frame_identity = geo.to_frame(origin, theta, np.array([1,1]))

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
    theta = np.pi / 2

    fig, ax = plt.subplots()
    ax.plot(0, 0, 'k+')
    ax.set_title(title)

    ax.set_ylim((0,10))
    ax.set_xlim((0,10))
    
    plot_frame(origin=origin, theta=theta, ax=ax)
    plt.show()

    