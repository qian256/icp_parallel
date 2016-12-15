#!/usr/bin/env python
"""
icp.py
- The base class of ICP: ICPBase
- Traditional ICP algorithm class: ICP

Author: Long Qian
Date: Dec 2016

"""
import numpy as np
from pointcloud import PointCloud
from utils import *
from plot import PointCloudPlotter



class ICPBase(object):
    """
    Base class for traditional ICP and paralleled ICP

    - Sketches the ICP algorithm skeleton
    - Define common attributes and functions
    """
    def __init__(self, src, dst, plot=True, reinit=False):
        """
        Initialize ICPBase class object

        :param src: source PointCloud object
        :param dst: destination PointCloud object
        :param plot: visualize the registration or not
        :param reinit: re-initialize the current PointCloud when local minima is encounted, or not
        """
        self.src = src
        self.dst = dst
        assert self.src.num == self.dst.num
        self.result = PointCloud(src)
        self.plot = plot
        self.reinit = reinit
        if self.plot: self.plotter = PointCloudPlotter(self.dst)



    def getPointCloudRegistration(self, target):
        """
        Compute the PointCloud registration with correspondence.
        It is same for both ICP and ICPParallel.

        :param target: the target PointCloud to register with
        :return: R: Rotation between PointCloud result and target
                T: Translation between PointCloud result and target
        """
        assert self.result and target
        assert self.result.num == target.num

        H = np.dot(self.result.normPoints.T, target.normPoints)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        # Consider reflection case
        if np.linalg.det(R) < 0:
            Vt[2,:] *= -1
            R = np.dot(Vt.T, U.T)
        # Raise a warning if the SVD is not correctly performed
        if abs(np.linalg.det(R) - 1.0) > 0.0001:
            Warning("Direct Point Cloud registration unstable!")
        T = target.center - self.result.center.dot(R.T)
        return R, T


    def computeCorrespondence(self):
        """
        Base function for finding correspondence.
        Assertion only. Detailed implementation varies.

        :return: total distance and updated PointCloud result
        """
        assert self.result and self.dst
        assert self.result.num == self.dst.num



    def solve(self):
        """
        Algorithmatic skeleton of ICP.
        It is common for both ICP and ICPParallel
        See PDF report or Besl's paper for more detials.

        :return: None
        """
        print 'Solve ICP with', self.__class__.__name__
        distanceThres, maxIteration, iteration = 0.001, 20, 0
        # Perform the initial computation of correspondence
        currentDistance, target = self.computeCorrespondence()
        print "Init ICP, distance: %f" % currentDistance
        # ICP loop
        while currentDistance > distanceThres and iteration < maxIteration:
            if self.plot: self.plotter.plotData(self.result)
            # Compute tranformation between self.result and self.target
            R, T = self.getPointCloudRegistration(target)

            # Appy transformation to self.result
            self.result.applyTransformation(R, T)

            # DEBUG
            if self.reinit:
                if np.amax(np.abs(R - np.identity(3))) < 0.000001 and np.amax(np.abs(T)) < 0.000001:
                    self.result.applyTransformation(getRandomRotation(), getRandomTranslation())

            # Compute point correspondence
            currentDistance, target = self.computeCorrespondence()

            # Update
            iteration += 1
            print "Iteration: %5d, with total distance: %f" % (iteration, currentDistance)






class ICP(ICPBase):
    """
    Traditional ICP algorithm
    - Inherited from ICPBase
    - No parallelism nor optimization
    """
    def __init__(self, src, dst, plot=True, reinit=False):
        """
        Initialize ICP
        :param src: source PointCloud object
        :param dst: destination PointCloud object
        :param plot: visualize the registration or not
        :param reinit: re-initialize the current PointCloud when local minima is encounted, or not
        """
        super(ICP, self).__init__(src, dst, plot, reinit)


    def getPointCloudRegistration(self, target):
        """
        Reuse ICPBase implementation
        :param target: target PointCloud to register with
        :return: Rotation and Translation from PointCloud result to target
        """
        return super(ICP, self).getPointCloudRegistration(target)



    def computeCorrespondence(self):
        """
        Compute point correspondence from source to destination PointCloud.
        Numpy is extensively used for computing and comparing distances.

        :return: total distance between PointCloud result and target
        """
        super(ICP, self).computeCorrespondence()

        indexArray = np.zeros(self.result.num, dtype=np.int)
        totalDistance = 0.0
        for resultIndex, resultValue in enumerate(self.result.points):
            minIndex, minDistance = -1, np.inf
            for dstIndex, dstValue in enumerate(self.dst.points):
                distance = np.linalg.norm(resultValue - dstValue)
                if distance < minDistance:
                    minDistance, minIndex = distance, dstIndex
            indexArray[resultIndex] = minIndex
            totalDistance += minDistance

        return totalDistance, PointCloud(self.dst.points[indexArray, 0:3])





if __name__ == '__main__':
    """
    When icp.py is used as main module,
    - Create a random PointClound pc1
    - Create a another PointCloud pc2 as a random transformation from pc1
    - Compute the ICP between them
    """
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-n', '--number', dest='pointNum', help='number of points in the point cloud', type='int', default=256)
    parser.add_option('-p', '--plot', dest='plot', action='store_true', help='visualize icp result', default=False)
    (options, args) = parser.parse_args()
    pc1 = PointCloud()
    pc1.initFromRand(options.pointNum, 0, 10, 10, 11, 20, 100)
    pc2 = PointCloud(pc1)
    pc2.applyTransformation()
    icpSolver = ICP(pc1, pc2, plot=options.plot)
    icpSolver.solve()



