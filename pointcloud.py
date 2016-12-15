#!/usr/bin/env python
"""
pointcloud.py
- The class for PointCloud

Author: Long Qian
Date: Dec 2016

"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from utils import getRandomTranslation, getRandomRotation


class PointCloud(object):
    """
    The Class of PointCloud

    - points: Numpy object, num*3, original point
    - normPoints: Numpy object, num*3, point location substracting mean value
    - num: the number of point
    - center: Numpy object, array of 3

    """
    def __init__(self, other=None):
        """
        Initialize PointCloud object

        :param other: when used as copy constructor, the copied PointCloud
        """
        self.points = None
        self.normPoints = None
        self.num = 0
        self.center = None
        if isinstance(other, PointCloud):
            # copy constructing
            self.initFromPointClound(other)
        elif isinstance(other, np.ndarray):
            # initialize from numpy array
            self.points = np.copy(other)
            self.num = other.shape[0]
            self.normalize()


    def normalize(self):
        """
        normalize the PointCloud, find the mean and create deMeaned points
        :return: None
        """
        assert self.num != 0
        self.center = np.mean(self.points, axis=0)
        self.normPoints = self.points - self.center


    def initFromRand(self, num, xmin, xmax, ymin, ymax, zmin, zmax):
        """
        Initialize from random point lists

        :param num: the number of points in the PointCloud
        :param xmin: minimum value of x axis
        :param xmax: maximum value of x axis
        :param ymin: minimum value of y axis
        :param ymax: maximum value of y axis
        :param zmin: minimum value of z axis
        :param zmax: maximum value of z axis
        :return: None
        """
        assert num > 0
        assert xmax > xmin
        assert ymax > ymin
        assert zmax > zmin
        self.points = np.random.rand(num, 3).astype(np.float32)
        self.points[:,0] = self.points[:,0] * (xmax-xmin) + xmin
        self.points[:,1] = self.points[:,1] * (ymax-ymin) + ymin
        self.points[:,2] = self.points[:,2] * (zmax-zmin) + zmin
        self.num = num
        self.normalize()


    def initFromPointClound(self, other):
        """
        Initialize from another PointCloud object

        :param other: the other PointCloud to be copied from
        :return: None
        """
        assert isinstance(other, PointCloud)
        self.points = np.copy(other.points)
        self.num = other.num
        self.normalize()


    def applyTransformation(self, R=None, T=None):
        """
        Apply rotation and translation on the current PointCloud

        :param R: Rotation, 3*3 Numpy array
        :param T: Translation, 3 element Numpy array
        :return: None
        """
        if not isinstance(R, np.ndarray): R = getRandomRotation()
        if not isinstance(T, np.ndarray): T = getRandomTranslation()
        self.points = self.points.dot(R.T) + T.T
        self.normalize()




def drawPointCloud(pc):
    """
    Visualize the PointCloud object with Matplotlib

    :param pc: PointCloud object
    :return: None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc.points[:,0], pc.points[:,1], pc.points[:,2], c='r', marker='.')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()



if __name__ == '__main__':
    """
    When pointcloud.py is used as main module,
    - Initialize a random PointCloud
    - Visualize it
    """
    pc = PointCloud()
    pc.initFromRand(10000, 0, 1, 10, 11, 0, 1)
    print pc.points
    drawPointCloud(pc)

