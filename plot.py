#!/usr/bin/env python
"""
plot.py
- The class for plotter of ICPBase object
- Plot the 3D point cloud, both current and target

Author: Long Qian
Date: Dec 2016

"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# save image locally or not
SAVE = False

class PointCloudPlotter(object):
    """
    PointCloudPlotter
    - Plots two point cloud: destination and current
    - Can be updated at runtime
    - Supports saving locally
    """
    def __init__(self, dst):
        """
        Init function of PointCloudPlotter
        :param dst: PointCloud object of destination
        """
        self.dst = dst
        self.ax = None
        self.resultScatter = None
        maxArray, minArray = np.amax(self.dst.points, axis=0), np.amin(self.dst.points, axis=0)
        self.dstxmax, self.dstxmin = maxArray[0], minArray[0]
        self.dstymax, self.dstymin = maxArray[1], minArray[1]
        self.dstzmax, self.dstzmin = maxArray[2], minArray[2]
        self.xmax, self.xmin = None, None
        self.ymax, self.ymin = None, None
        self.zmax, self.zmin = None, None
        self.seq = 0

    def updateXYZLim(self, result):
        """
        Function to dynamically update the XYZ axis upper and lower limits
        :param result: PointCloud object of result
        :return: None
        """
        maxArray, minArray = np.amax(result.points, axis=0), np.amin(result.points, axis=0)
        self.xmax = max(self.dstxmax, maxArray[0])
        self.xmin = min(self.dstxmin, minArray[0])
        self.ymax = max(self.dstymax, maxArray[1])
        self.ymin = min(self.dstymin, minArray[1])
        self.zmax = max(self.dstzmax, maxArray[2])
        self.zmin = min(self.dstzmin, minArray[2])
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.ymin, self.ymax)
        self.ax.set_zlim(self.zmin, self.zmax)


    def plotData(self, result):
        """
        Plot function
        :param result: PointCloud object of result
        :return: None
        """

        # Ignore warnings
        import warnings
        warnings.filterwarnings("ignore")

        if not self.ax:
            # First iteration
            plt.ion()
            fig = plt.figure()
            self.ax = fig.add_subplot(111, projection='3d')
            self.resultScatter = self.ax.scatter(result.points[:,0], result.points[:,1], result.points[:,2], c='r', marker='o')
            self.ax.scatter(self.dst.points[:,0], self.dst.points[:,1], self.dst.points[:,2], c='b', marker='^')
            self.ax.set_xlabel('X Label')
            self.ax.set_ylabel('Y Label')
            self.ax.set_zlabel('Z Label')
            self.updateXYZLim(result)
            plt.draw()
            plt.pause(0.001)
            if SAVE: plt.savefig(str(self.seq) + '.png')
            self.seq += 1
        else:
            # other iterations
            self.resultScatter.remove()
            self.resultScatter = self.ax.scatter(result.points[:,0], result.points[:,1], result.points[:,2], c='r', marker='o')
            self.updateXYZLim(result)
            plt.draw()
            plt.pause(0.001)
            if SAVE: plt.savefig(str(self.seq) + '.png')
            self.seq += 1










