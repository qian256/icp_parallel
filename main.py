#!/usr/bin/env python
"""
main.py
- The main module for CUDA ICP Project
- Compares normal ICP and paralleled ICP

Author: Long Qian
Date: Dec 2016

"""
from icp import ICP
from icp_parallel import ICPParallel
from pointcloud import PointCloud


if __name__ == '__main__':
    """
    Main entry for main.py
    - Parse options
    - Run normal ICP
    - Run paralleled ICP
    """
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-n', '--number', dest='pointNum', help='number of points in the point cloud', type='int', default=256)
    parser.add_option('-p', '--plot', dest='plot', action='store_true', help='visualize icp result', default=False)
    parser.add_option('-c', '--core', dest='coreNum', help='number of threads used in the cuda', type='int', default=0)
    (options, args) = parser.parse_args()
    pc1 = PointCloud()
    pc1.initFromRand(options.pointNum, 0, 1, 10, 20, 0, 1)
    pc2 = PointCloud(pc1)
    pc2.applyTransformation(None, None)
    icpSolver = ICP(pc1, pc2, plot=options.plot)
    icpSolver.solve()
    print
    icpParallelSolver = ICPParallel(pc1, pc2, numCore=options.coreNum, plot=options.plot)
    icpParallelSolver.solve()

