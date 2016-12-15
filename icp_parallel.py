#!/usr/bin/env python
"""
icp_parallel.py
- Paralleled ICP algorithm class: ICPParallel

Author: Long Qian
Date: Dec 2016

"""
import numpy as np
from pointcloud import PointCloud
from utils import *
from plot import PointCloudPlotter
from icp import ICPBase

# PyCuda related modules
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule



class ICPParallel(ICPBase):
    """
    Paralleled ICP class
    - Inherited from ICPBase
    - Use PyCuda for parallelization
    """
    def __init__(self, src, dst, numCore=0, plot=True, reinit=False):
        """
        Reuse ICPBase initializer, and initialize CUDA environment

        :param src: source PointCloud object
        :param dst: destination PointCloud object
        :param numCore: the number of threads to be used to parallelize the algorithm
        :param plot: visualize the registration or not
        :param reinit: re-initialize the current PointCloud when local minima is encounted, or not
        """
        super(ICPParallel, self).__init__(src, dst, plot, reinit)
        # CUDA related utilities
        self.computeCorrespondenceCuda = None
        self.distances_gpu = None
        self.numCore = self.src.num if numCore <= 0 else numCore
        print 'ICP Parallel cuda cores set to', self.numCore
        self.initCuda()


    def initCuda(self):
        """
        Initialize CUDA environment
        - Create CUDA C program, and compile it
        - Copy destination PointCloud to CUDA global and constant value
        - Create handler of find_closest function in CUDA

        :return: None
        """
        mod = SourceModule("""
        #define ROW (""" + str(self.src.num) + """)
        #define OFFSET (""" + str(self.numCore) + """)
        __constant__ float dst[ROW][3];

        __global__ void get_dst(float* ret){
            int idx = threadIdx.x;
            ret[idx*3] = dst[idx][0];
            ret[idx*3+1] = dst[idx][1];
            ret[idx*3+2] = dst[idx][2];
        }

        __global__ void find_closest(float* result, float *ret, float* distances)
        {
            for (int idx = threadIdx.x; idx < ROW; idx += OFFSET) {
                float x_src = result[idx*3];
                float y_src = result[idx*3+1];
                float z_src = result[idx*3+2];
                float x_dst, y_dst, z_dst, minDist, dist;
                int minIdx = -1;
                for ( int i = 0; i < ROW; i++){
                    x_dst = dst[i][0];
                    y_dst = dst[i][1];
                    z_dst = dst[i][2];
                    dist = (x_src-x_dst) * (x_src-x_dst) + (y_src-y_dst) * (y_src-y_dst) + (z_src-z_dst) * (z_src-z_dst);
                    if ( dist < minDist || minIdx < 0 ) {
                        minDist = dist;
                        minIdx = i;
                    }
                }
                ret[idx*3] = dst[minIdx][0];
                ret[idx*3+1] = dst[minIdx][1];
                ret[idx*3+2] = dst[minIdx][2];
                distances[idx] = sqrt(minDist);
            }
        }
        """)
        dstCuda, _ = mod.get_global('dst')
        assert self.dst.points.dtype == np.float32
        cuda.memcpy_htod(dstCuda, self.dst.points)
        distances = np.zeros(self.src.num, dtype=np.float32)
        self.distances_gpu = gpuarray.to_gpu(distances)
        self.computeCorrespondenceCuda = mod.get_function('find_closest')
        getDstCuda = mod.get_function('get_dst')
        # temp = np.zeros([self.src.num, 3], dtype=np.float32)
        # getDstCuda(cuda.Out(temp), block=(self.src.num, 1, 1))
        # print 'dst in cuda set to be:'
        # print temp


    def getPointCloudRegistration(self, target):
        """
        Reuse ICPBase implementation
        :param target: target PointCloud to register with
        :return: Rotation and Translation from PointCloud result to target
        """
        return super(ICPParallel, self).getPointCloudRegistration(target)


    def computeCorrespondence(self):
        """
        Compute point correspondence from result PointCloud to dst.
        CUDA function and summation reduction is called here.

        :return: total distance and matrix with point correspondence
        """
        super(ICPParallel, self).computeCorrespondence()

        target = np.zeros([self.src.num, 3], dtype=np.float32)
        self.computeCorrespondenceCuda(cuda.In(self.result.points), cuda.Out(target), self.distances_gpu, block=(self.numCore, 1, 1))

        return gpuarray.sum(self.distances_gpu).get(), PointCloud(target)



if __name__ == '__main__':
    """
    When icp_parallel.py is used as main module
    - Create a random PointClound pc1
    - Create a another PointCloud pc2 as a random transformation from pc1
    - Compute the ICP between them, using paralleled ICP
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
    pc2.applyTransformation()
    icpSolver = ICPParallel(pc1, pc2, numCore=options.coreNum, plot=options.plot)
    icpSolver.solve()



