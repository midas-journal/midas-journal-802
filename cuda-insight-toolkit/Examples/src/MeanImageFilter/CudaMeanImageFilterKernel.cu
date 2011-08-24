/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaMeanImageFilterKernel.cu
  Language:  CUDA

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
/** \class CudaMeanImageFilterKernel.cu
 * \brief Cuda kernel code
 * \author Phillip Ward, Luke Parkinson, Daniel Micevski, Christopher
 * Share, Victorian Partnership for Advanced Computing (VPAC). 
 * Richard Beare, Monash University
 */

#include "EclipseCompat.h"
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cutil.h>
#include "CudaNeighborhoodFunctions.cu"

// Pointer to shared memory window
extern __shared__ float sharedMem[];
// Pointer to 2D Texture
texture<float, 2> texRef2D;
// Pointer to 3D Texture
texture<float, 3> texRef3D;

template<class T>
__global__ void CudaMeanImageFilterKernelShared2D(T *output, int2 imageDim,
		int2 radius, int N) {

	// Compute pixel coordinates of thread
	int2 pixel = make_int2((blockIdx.x * blockDim.x) + threadIdx.x, (blockIdx.y
			* blockDim.y) + threadIdx.y);

	// Compute window size for shared memory
	int2 window = make_int2((2 * radius.x) + blockDim.x, (2 * radius.y)
			+ blockDim.y);

	// Populate shared memory window
	PopulateSharedMem2D(sharedMem, texRef2D, imageDim, radius, pixel, window);

	T mean = 0;
	// Sync threads to ensure window is completely populated
	__syncthreads();

	// Returns each value in neighborhood
	for (int j = 0; j <= 2 * radius.y; ++j) {
		for (int i = 0; i <= 2 * radius.x; ++i) {
			// Critical section here
			mean += GetSharedMemValue2D(sharedMem, i, j, imageDim, radius,
					pixel, window);
		}
	}

	// Sync threads before doing writes to global
	if (pixel.y < imageDim.y && pixel.x < imageDim.x) {
		// Write value to output here
		output[pixel.y * imageDim.x + pixel.x] = mean / ((2 * radius.x + 1)
				* (2 * radius.y + 1));
	}
}

template<class T>
__global__ void CudaMeanImageFilterKernelGlobal2D(T *output, int2 imageDim,
		int2 radius, int N, int offset) {

	// Compute threads linear position
	int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

	if (idx < N) {
		// Compute pixel coordinates of thread
		int2 pixel = make_int2(idx % imageDim.x, idx / imageDim.x);
		T mean = 0;

		// Returns each value in neighborhood
		for (int j = 0; j <= 2 * radius.y; j++) {
			for (int i = 0; i <= 2 * radius.x; i++) {
				// Add critical section here
				mean += GetGlobalValue2D(texRef2D, i, j, imageDim, radius,
						pixel);
			}
		}

		// Sync before writing back to global
		__syncthreads();

		// Write output
		output[idx] = mean / ((2 * radius.x + 1) * (2 * radius.y + 1));
	}
}

template<class T>
__global__ void CudaMeanImageFilterKernelGlobal3D(T *output, int3 imageDim,
		int3 radius, int N, int offset) {

	// Compute threads linear position
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

	if (idx < N) {

		// Compute pixel coordinates of thread
		int3 pixel = make_int3(0, 0, 0);
		pixel.x = idx % imageDim.x;
		pixel.y = (idx / imageDim.x) % imageDim.y;
		pixel.z = idx / (imageDim.y * imageDim.x);

		T mean = 0;

		// Returns each value in neighborhood
		for (int k = 0; k <= 2 * radius.z; k++) {
			for (int j = 0; j <= 2 * radius.y; j++) {
				for (int i = 0; i <= 2 * radius.x; i++) {
					// Add critical section here
					mean += GetGlobalValue3D(texRef3D, i, j, k, imageDim,
							radius, pixel);
				}
			}
		}

		// Sync before writing back to global
		__syncthreads();

		// Write output
		output[idx] = mean / ((2 * radius.x + 1) * (2 * radius.y + 1) * (2
				* radius.z + 1));
	}
}
template <class T, class S>
void CudaMeanImageFilterKernelFunction(const T* input, S *output,
		unsigned int imageDimX, unsigned int imageDimY, unsigned int imageDimZ,
		unsigned int radiusX, unsigned int radiusY, unsigned int radiusZ,
		unsigned int N) 
{

	// Get device properties to compute block and grid size later
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);

	// 3D Image
	if (imageDimZ > 1) {
		int3 radius = make_int3(radiusX, radiusY, radiusZ);
		int3 imageDim = make_int3(imageDimX, imageDimY, imageDimZ);

		// Allocate Cuda Array
		cudaArray *texArray = 0;
		cudaChannelFormatDesc cf = cudaCreateChannelDesc<T> ();
		cudaExtent const ext = { imageDim.x, imageDim.y, imageDim.z };
		CUDA_SAFE_CALL(cudaMalloc3DArray(&texArray, &cf, ext));
		CUT_CHECK_ERROR("Malloc 3D Array Failed\n");

		// Bind to Texture
		CUDA_SAFE_CALL(cudaBindTextureToArray(texRef3D, texArray, cf));
		CUT_CHECK_ERROR("Bind Texture To Array Failed\n");

		// Copy Linear Device Memory into Cuda Array
		cudaMemcpy3DParms copyParams = { 0 };
		copyParams.srcPtr = make_cudaPitchedPtr(const_cast<T *> (input),
				imageDim.x * sizeof(T), imageDim.x, imageDim.y);
		copyParams.dstArray = texArray;
		copyParams.kind = cudaMemcpyDeviceToDevice;
		copyParams.extent = ext;
		CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
		CUT_CHECK_ERROR("Memcpy Device -> Array Failed\n");

		// specify mutable texture reference parameters
		texRef3D.normalized = 0;
		texRef3D.filterMode = cudaFilterModePoint;
		texRef3D.addressMode[0] = cudaAddressModeClamp;
		texRef3D.addressMode[1] = cudaAddressModeClamp;
		texRef3D.addressMode[2] = cudaAddressModeClamp;

		// Calculate block size based on register limit
		int blockSize = 64;//devProp.maxThreadsPerBlock;

		while (blockSize * 19 > devProp.regsPerBlock) {
			blockSize /= 2;
		}

		// Calculate Grid Size and Kernel Passes Required
		int nBlocks = N / blockSize + (N % blockSize == 0 ? 0 : 1);
		int runs = nBlocks / (devProp.maxGridSize[0]) + (nBlocks
				% (devProp.maxGridSize[0]) == 0 ? 0 : 1);

		int i = 1;
		for (; i * 1000 < radius.x * radius.y * radius.z; ++i)
			;
		runs *= i;

		nBlocks /= runs;

		// Execute Kernel Passes
		for (int i = 0; i < runs; i++) {
			int offset = i * nBlocks * blockSize;
			CudaMeanImageFilterKernelGlobal3D <<< nBlocks, blockSize >>>
			(output, imageDim, radius, (int)N, offset);
			cudaThreadSynchronize();
		}

		//printf("Kernel Error: %s\n", cudaGetErrorString(cudaGetLastError()));

		// Free Array and Unbind Texture
		cudaFreeArray(texArray);
		cudaUnbindTexture(texRef3D);

		// Return pointer to the output
	}
	// 2D Image
	else {

		int2 radius = make_int2(radiusX, radiusY);
		int2 imageDim2 = make_int2(imageDimX, imageDimY);

		// set up the CUDA array
		cudaChannelFormatDesc cf = cudaCreateChannelDesc<T> ();
		cudaArray *texArray = 0;
		cudaMallocArray(&texArray, &cf, imageDim2.x, imageDim2.y);
		cudaMemcpyToArray(texArray, 0, 0, input, sizeof(T) * N,
				cudaMemcpyDeviceToDevice);

		// specify mutable texture reference parameters
		texRef2D.normalized = 0;
		texRef2D.filterMode = cudaFilterModePoint;
		texRef2D.addressMode[0] = cudaAddressModeClamp;
		texRef2D.addressMode[1] = cudaAddressModeClamp;

		// bind texture reference to array
		cudaBindTextureToArray(texRef2D, texArray);

		// Calculate block size based on Maximum Registers per Block
		int blockSize = devProp.maxThreadsPerBlock;
		while (blockSize * 19 > devProp.regsPerBlock) {
			blockSize /= 2;
		}

		// Calculate Grid Size and Kernel Passes Required
		int nBlocks = N / blockSize + (N % blockSize == 0 ? 0 : 1);
		int runs = nBlocks / devProp.maxGridSize[0] + (nBlocks
				% devProp.maxGridSize[0] == 0 ? 0 : 1);

		int i = 1;
		for (; i * 1000 < radius.x * radius.y; ++i)
			;
		runs *= i;
		nBlocks /= runs;

		// Execute Kernel Passes
		for (int i = 0; i < runs; i++) {
			int offset = i * nBlocks * blockSize;
			CudaMeanImageFilterKernelGlobal2D <<< nBlocks, blockSize >>> (output, imageDim2, radius, (int)N, offset);
			cudaThreadSynchronize();
		}

		//printf("Kernel Error: %s\n", cudaGetErrorString(cudaGetLastError()));

		cudaFreeArray(texArray);
		cudaUnbindTexture(texRef2D);

		// Return pointer to the output
	}
}

#define THISTYPE float
template void CudaMeanImageFilterKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, THISTYPE * output, 
							 unsigned int imageDimX, unsigned int imageDimY, 
							 unsigned int imageDimZ,
							 unsigned int radiusX, unsigned int radiusY, 
							 unsigned int radiusZ,
							 unsigned int N);
#undef THISTYPE
#define THISTYPE int
template void CudaMeanImageFilterKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, THISTYPE * output, 
							 unsigned int imageDimX, unsigned int imageDimY, 
							 unsigned int imageDimZ,
							 unsigned int radiusX, unsigned int radiusY, 
							 unsigned int radiusZ,
							 unsigned int N);
#undef THISTYPE

#define THISTYPE short
template void CudaMeanImageFilterKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, THISTYPE * output, 
							 unsigned int imageDimX, unsigned int imageDimY, 
							 unsigned int imageDimZ,
							 unsigned int radiusX, unsigned int radiusY, 
							 unsigned int radiusZ,
							 unsigned int N);
#undef THISTYPE

#define THISTYPE unsigned char
template void CudaMeanImageFilterKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, THISTYPE * output, 
							 unsigned int imageDimX, unsigned int imageDimY, 
							 unsigned int imageDimZ,
							 unsigned int radiusX, unsigned int radiusY, 
							 unsigned int radiusZ,
							 unsigned int N);
#undef THISTYPE
