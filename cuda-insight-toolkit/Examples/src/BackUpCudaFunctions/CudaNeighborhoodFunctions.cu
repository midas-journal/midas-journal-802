/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaNeighborhoodFunctions.cu
  Language:  CUDA

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
/** \class CudaNeighborhoodFunctions.cu
 * \brief Cuda kernel code
 * \author Phillip Ward, Luke Parkinson, Daniel Micevski, Christopher
 * Share, Victorian Partnership for Advanced Computing (VPAC). 
 * Richard Beare, Monash University
 */

/*
 * File Name:    Cuda NeighboorHood Functions
 *
 * Author:        Phillip Ward
 * Creation Date: Monday, January 18 2010, 10:00 
 * Last Modified: Wednesday, February 23 2010, 16:35
 * 
 * File Description:
 *
 * Used with caution, block size must be monitored for registers
 *  and shared memory limitations.
 *
 * imageDim refers to the dimensions of the image.
 * radius refers to the radius in each dimensions.
 * pixel refers to the coordinates of the pixel that thread
 * corresponds to.
 * window refers to the window of shared memory. It is typically
 * 2 * radius + blockDim in each dimension.
 *
 *
 *
 */
#include "EclipseCompat.h"
#include <stdio.h>
#include <cuda.h>
#include <cutil.h>

template<class T>
__device__ void PopulateSharedMem2D(T *sharedMem, texture<T, 2> texRef,
		int2 imageDim, int2 radius, int2 pixel, int2 window) {
	for (int j = 0; j < window.y; j += blockDim.y) {
		for (int i = 0; i < window.x; i += blockDim.x) {
			if ((threadIdx.x + i) < window.x && (threadIdx.y + j) < window.y) {
				if ((pixel.y - radius.y + j) > -1 && (pixel.y - radius.y + j)
						< imageDim.y && (pixel.x - radius.x + i) > -1
						&& (pixel.x - radius.x + i) < imageDim.x) {
					sharedMem[threadIdx.x + i + ((threadIdx.y + j) * window.x)]
							= tex2D(texRef, pixel.x - radius.x + i, pixel.y
									- radius.y + j);
				}
			}
		}
	}
}

template<class T>
__device__ void PopulateSharedMem3D(T *sharedMem, texture<T, 3> texRef,
		int3 imageDim, int3 radius, int3 pixel, int3 window) {
	for (int k = 0; k < window.z; k += blockDim.z) {
		for (int j = 0; j < window.y; j += blockDim.y) {
			for (int i = 0; i < window.x; i += blockDim.x) {
				if ((threadIdx.x + i) < window.x && (threadIdx.y + j)
						< window.y && (threadIdx.z + k) < window.z) {
					if ((pixel.z - radius.z + k) > -1 && (pixel.z - radius.z
							+ k) < imageDim.z && (pixel.y - radius.y + j) > -1
							&& (pixel.y - radius.y + j) < imageDim.y
							&& (pixel.x - radius.x + i) > -1 && (pixel.x
							- radius.x + i) < imageDim.x) {
						sharedMem[threadIdx.x + i + ((threadIdx.y + j)
								* window.x) + ((threadIdx.z + k) * window.y
								* window.x)] = tex3D(texRef, pixel.x - radius.x
								+ i, pixel.y - radius.y + j, pixel.z - radius.z
								+ k);
					}
				}
			}
		}
	}
}

template<class T>
__device__ T GetSharedMemValue2D(T* sharedMem, int i, int j, int2 imageDim,
		int2 radius, int2 pixel, int2 window) {

	int x = threadIdx.x + i;
	int y = threadIdx.y + j;

	if (pixel.y + radius.y >= imageDim.y || pixel.x + radius.x >= imageDim.x
			|| pixel.x - radius.x < 0 || pixel.y - radius.y < 0) {
		if (pixel.y + j - radius.y >= imageDim.y) {
			y = imageDim.y - 1 - pixel.y + threadIdx.y + radius.y;
		} else if (pixel.y + j - radius.y < 0) {
			y = threadIdx.y + radius.y - pixel.y;
		}
		if (pixel.x + i - radius.x >= imageDim.x) {
			x = imageDim.x - 1 - pixel.x + threadIdx.x + radius.x;
		} else if (pixel.x + i - radius.x < 0) {
			x = threadIdx.x + radius.x - pixel.x;
		}
	}

	return sharedMem[x + (y * window.x)];
}

template<class T>
__device__ T GetSharedMemValue3D(T* sharedMem, int i, int j, int k,
		int3 imageDim, int3 radius, int3 pixel, int3 window) {

	int x = threadIdx.x + i;
	int y = threadIdx.y + j;
	int z = threadIdx.z + k;

	if (pixel.z + radius.z >= imageDim.z || pixel.y + radius.y >= imageDim.y
			|| pixel.x + radius.x >= imageDim.x || pixel.z - radius.z < 0
			|| pixel.y - radius.y < 0 || pixel.x - radius.x < 0) {
		if (pixel.z + k - radius.z >= imageDim.z) {
			z = imageDim.z - 1 - pixel.z + threadIdx.z + radius.z;
		} else if (pixel.z + k - radius.z < 0) {
			z = threadIdx.z + radius.z - pixel.z;
		}
		if (pixel.y + j - radius.y >= imageDim.y) {
			y = imageDim.y - 1 - pixel.y + threadIdx.y + radius.y;
		} else if (pixel.y + j - radius.y < 0) {
			y = threadIdx.y + radius.y - pixel.y;
		}
		if (pixel.x + i - radius.x >= imageDim.x) {
			x = imageDim.x - 1 - pixel.x + threadIdx.x + radius.x;
		} else if (pixel.x + i - radius.x < 0) {
			x = threadIdx.x + radius.x - pixel.x;
		}
	}

	return sharedMem[x + ((y + (z * window.y)) * window.x)];
}

template<class T>
__device__ T GetGlobalValue2D(texture<T, 2> texRef, int i, int j,
		int2 imageDim, int2 radius, int2 pixel) {
	int x = pixel.x - radius.x + i;
	int y = pixel.y - radius.y + j;
	if (y >= imageDim.y)
		y = imageDim.y - 1;
	if (x >= imageDim.x)
		x = imageDim.x - 1;
	if (y < 0)
		y = 0;
	if (x < 0)
		x = 0;

	return tex2D(texRef, x, y);
}

template<class T>
__device__ T GetGlobalValue3D(texture<T, 3> texRef, int i, int j, int k,
		int3 imageDim, int3 radius, int3 pixel) {

	int x = pixel.x - radius.x + i;
	int y = pixel.y - radius.y + j;
	int z = pixel.z - radius.z + k;

	if (z >= imageDim.z)
		z = imageDim.z - 1;
	if (y >= imageDim.y)
		y = imageDim.y - 1;
	if (x >= imageDim.x)
		x = imageDim.x - 1;
	if (z < 0)
		z = 0;
	if (y < 0)
		y = 0;
	if (x < 0)
		x = 0;

	return tex3D(texRef, x, y, z);
}

template<class T>
__device__ T GetGlobalValue3D(T * input, int i, int j, int k, int3 imageDim,
		int3 radius, int3 pixel) {

	int x = pixel.x - radius.x + i;
	int y = pixel.y - radius.y + j;
	int z = pixel.z - radius.z + k;

	if (z >= imageDim.z)
		z = imageDim.z - 1;
	else if (z < 0)
		z = 0;

	if (y >= imageDim.y)
		y = imageDim.y - 1;
	else if (y < 0)
		y = 0;

	if (x >= imageDim.x)
		x = imageDim.x - 1;
	else if (x < 0)
		x = 0;

	return input[x + (y * imageDim.x) + (z * imageDim.y * imageDim.x)];
}
