#!/usr/bin/env python
# coding : utf-8

import numpy as np

# Convention :
# Data vectors will be represented by as such : (data, dimension)
# For an explaination, see : https://github.com/Lasagne/Lasagne/issues/30

def cartesian_to_polar(x, y):
    theta = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    return r, theta

def polar_to_cartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def to_frame(origin, basis, points):
    """
    Projects a (N, 2) vector of cartesian coordinates into a frame defined by its origin and an orthogonal basis.
    Note that it is important for the basis to be orthogonal.
    
    Practically, the origin is usually some translation vector (displacement) while the basis is a rotation matrix (orientation).
    Finally, the canonical base can be thought as the world frame or a any other reference frame.


    :param origin: a (1,2) vector defining the origin of the basis.
    :param basis: a (2,2) matrix defining the basis.
    :param points: a (N,2) vector to be projected.

    :return: a (N,2) vector of cartesian coordinates projected in the target basis.

    :type origin: a numpy nd-array.
    :type basis: a numpy nd-array.
    """
    projected = basis.dot(points) + origin
    return projected

def from_frame(origin, basis, points):
    """
    Projects a (N, 2) vector of cartesian coordinates from a frame defined by its origin and an orthogonal basis, back to the canonical base.
    Note that it is important for the basis to be orthogonal (for we use the transpose as the inverse).
    
    Practically, the origin is usually some translation vector (displacement) while the basis is a rotation matrix (orientation).
    Finally, the canonical base can be thought as the world frame or a any other reference frame.

    :param origin: a (1,2) vector defining the origin of the basis.
    :param basis: a (2,2) matric defining the basis.

    :return: a (N,2) vector of cartesian coordinates projected back in the canonical basis.

    :type origin: a numpy nd-array.
    :type basis: a numpy nd-array.
    """
    projected = basis.T.dot(points - origin)
    return projected

if __name__ == '__main__':
    translate = np.array([3, 7])
    base = np.array([[-1, 0],[0,-1]])
    unit = np.array([1, 1])
    
    projected = to_frame(translate, base, unit)
    np.testing.assert_array_equal(projected, np.array([2, 6]), 'to_frame did not project correctly')

    projected_back = from_frame(translate, base, projected)
    np.testing.assert_array_equal(projected_back, unit, 'from_frame did not project correctly')
