/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaGrayscaleDilateImageFilterKernel.cu
  Language:  CUDA

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
/** \class CudaGrayscaleDilateImageFilterKernel.cu
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

#define KTYPE unsigned char

// Pointer to 2D Texture
texture<KTYPE, 2> texRef2DKernel;
texture<float, 2> texRef2DInput;
// Pointer to 3D Texture
texture<KTYPE, 3> texRef3DKernel;
texture<float, 3> texRef3DInput;

template<class T>
__global__ void CudaGrayscaleDilateImageFilterKernelGlobal2D(T *output,
							     int2 imageDim, int2 radius, int2 kernelDim, unsigned long N, int offset) 
{

  // Compute threads linear position
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

  if (idx < N) 
    {
    // Compute pixel coordinates of thread
    int2 pixel = make_int2(idx % imageDim.x, idx / imageDim.x);
    T max = GetGlobalValue2D(texRef2DInput, radius.x+1, radius.y+1, imageDim,
			     radius, pixel);
    T temp;

    // Returns each value in neighborhood
    for (int j = 0; j < kernelDim.y; j++) 
      {
      for (int i = 0; i < kernelDim.x; i++) 
	{
	// Add critical section here
	if (tex2D(texRef2DKernel, i, j) > 0) {
	temp = GetGlobalValue2D(texRef2DInput, i, j, imageDim,
				radius, pixel);
	max = (temp > max) ? temp : max;
	}
	}
      }

    // Sync before writing back to global
    __syncthreads();

    // Write output
    output[idx] = max;
    }
}

template<class T>
__global__ void CudaGrayscaleDilateImageFilterKernelGlobal3D(T *output,
							     int3 imageDim, int3 radius, 
							     int3 kernelDim, unsigned long N, int offset) 
{

  // Compute threads linear position
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

  if (idx < N) 
    {

    // Compute pixel coordinates of thread
    int3 pixel = make_int3(0, 0, 0);
    pixel.x = idx % imageDim.x;
    pixel.y = (idx / imageDim.x) % imageDim.y;
    pixel.z = idx / (imageDim.y * imageDim.x);

    T max = GetGlobalValue3D(texRef3DInput, radius.x+1, radius.y+1, radius.z+1,
			     imageDim, radius, pixel);
    T temp;

    // Returns each value in neighborhood
    for (int k = 0; k < kernelDim.z; k++) 
      {
      for (int j = 0; j < kernelDim.y; j++) 
	{
	for (int i = 0; i < kernelDim.x; i++) 
	  {
	  // Add critical section here
	  if (tex3D(texRef3DKernel, i, j, k) > 0) 
	    {
	    temp = GetGlobalValue3D(texRef3DInput, i, j, k,
				    imageDim, radius, pixel);
	    max = (temp > max) ? temp : max;
	    }
	  }
	}
      }

    // Sync before writing back to global
    __syncthreads();

    // Write output
    output[idx] = max;
    }
}

template <class T, class S, class K> 
void CudaGrayscaleDilateImageFilterKernelFunction(const T* input, S* output,
						  const unsigned long * iDim, const unsigned long * r,
						  const K * kernel, const unsigned long * kDim,
						  const K zero, unsigned long D, unsigned long N)
{	
// Get device properties to compute block and grid size later
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, 0);


  // 3D Image
  if (D == 3) 
    {

    int3 radius = make_int3(r[0], r[1], r[2]);
    int3 imageDim = make_int3(iDim[0], iDim[1], iDim[2]);
    int3 kernelDim = make_int3(kDim[0], kDim[1], kDim[2]);

    // Allocate Cuda Arrays and Bind Textures

    // Kernel Array
    cudaArray *texKernelArray = 0;
    cudaChannelFormatDesc cfka = cudaCreateChannelDesc<K> ();
    cudaExtent const extKernelArray = { kernelDim.x, kernelDim.y,
					kernelDim.z };
    CUDA_SAFE_CALL(cudaMalloc3DArray(&texKernelArray, &cfka, extKernelArray));
    CUT_CHECK_ERROR("Malloc 3D Kernel Array Failed\n");

    // Bind to Texture
    CUDA_SAFE_CALL(cudaBindTextureToArray(texRef3DKernel, texKernelArray, cfka));
    CUT_CHECK_ERROR("Bind Texture To Kernel Array Failed\n");

    // Copy Linear Device Memory into Cuda Array
    cudaMemcpy3DParms kernelCopyParams = { 0 };
    kernelCopyParams.srcPtr = make_cudaPitchedPtr((void *)const_cast<K*>(kernel), kernelDim.x * sizeof(K), kernelDim.x, kernelDim.y);
    kernelCopyParams.dstArray = texKernelArray;
    kernelCopyParams.kind = cudaMemcpyHostToDevice;
    kernelCopyParams.extent = extKernelArray;
    CUDA_SAFE_CALL(cudaMemcpy3D(&kernelCopyParams));
    CUT_CHECK_ERROR("Memcpy Device -> Kernel Array Failed\n");

    // specify mutable texture reference parameters
    texRef3DKernel.normalized = 0;
    texRef3DKernel.filterMode = cudaFilterModePoint;
    texRef3DKernel.addressMode[0] = cudaAddressModeClamp;
    texRef3DKernel.addressMode[1] = cudaAddressModeClamp;
    texRef3DKernel.addressMode[2] = cudaAddressModeClamp;

    // Allocate Cuda Array
    cudaArray *texInputArray = 0;
    cudaChannelFormatDesc cfia = cudaCreateChannelDesc<T> ();
    cudaExtent const extInputArray = { imageDim.x, imageDim.y, imageDim.z };
    CUDA_SAFE_CALL(cudaMalloc3DArray(&texInputArray, &cfia, extInputArray));
    CUT_CHECK_ERROR("Malloc 3D Input Array Failed\n");

    // Bind to Texture
    CUDA_SAFE_CALL(cudaBindTextureToArray(texRef3DInput, texInputArray, cfia));
    CUT_CHECK_ERROR("Bind Texture To Kernel Array Failed\n");

    // Copy Linear Device Memory into Cuda Array
    cudaMemcpy3DParms inputCopyParams = { 0 };
    inputCopyParams.srcPtr = make_cudaPitchedPtr(
      const_cast<T *> (input), imageDim.x * sizeof(T),
      imageDim.x, imageDim.y);
    inputCopyParams.dstArray = texInputArray;
    inputCopyParams.kind = cudaMemcpyDeviceToDevice;
    inputCopyParams.extent = extInputArray;
    CUDA_SAFE_CALL(cudaMemcpy3D(&inputCopyParams));
    CUT_CHECK_ERROR("Memcpy Device -> Input Array Failed\n");

    // specify mutable texture reference parameters
    texRef3DInput.normalized = 0;
    texRef3DInput.filterMode = cudaFilterModePoint;
    texRef3DInput.addressMode[0] = cudaAddressModeClamp;
    texRef3DInput.addressMode[1] = cudaAddressModeClamp;
    texRef3DInput.addressMode[2] = cudaAddressModeClamp;

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
    CudaGrayscaleDilateImageFilterKernelGlobal3D <<< nBlocks, blockSize >>>
      (output, imageDim, radius, kernelDim, N, offset);
    cudaThreadSynchronize();
    }



    // Free Array and Unbind Texture
    cudaFreeArray(texKernelArray);
    cudaFreeArray(texInputArray);
    cudaUnbindTexture(texRef3DKernel);
    cudaUnbindTexture(texRef3DInput);
    }
  // 2D Image
  else 
    {

    int2 radius = make_int2(r[0], r[1]);
    int2 imageDim = make_int2(iDim[0], iDim[1]);
    int2 kernelDim = make_int2(kDim[0], kDim[1]);

    // Set up Cuda Arrays and Bind to Textures

    // set up the CUDA kernel array
    cudaChannelFormatDesc cfka = cudaCreateChannelDesc<K> ();
    cudaArray *texKernelArray = 0;
    cudaMallocArray(&texKernelArray, &cfka, kernelDim.x, kernelDim.y);
    cudaMemcpyToArray(texKernelArray, 0, 0, kernel, sizeof(K) * kernelDim.x * kernelDim.y,
		      cudaMemcpyHostToDevice);

    // specify mutable texture reference parameters
    texRef2DKernel.normalized = 0;
    texRef2DKernel.filterMode = cudaFilterModePoint;
    texRef2DKernel.addressMode[0] = cudaAddressModeClamp;
    texRef2DKernel.addressMode[1] = cudaAddressModeClamp;

    // bind texture reference to array
    cudaBindTextureToArray(texRef2DKernel, texKernelArray);

    // set up the CUDA array
    cudaChannelFormatDesc cfia = cudaCreateChannelDesc<T> ();
    cudaArray *texInputArray = 0;
    cudaMallocArray(&texInputArray, &cfia, imageDim.x, imageDim.y);
    cudaMemcpyToArray(texInputArray, 0, 0, input, sizeof(T) * N,
		      cudaMemcpyDeviceToDevice);

    // specify mutable texture reference parameters
    texRef2DInput.normalized = 0;
    texRef2DInput.filterMode = cudaFilterModePoint;
    texRef2DInput.addressMode[0] = cudaAddressModeClamp;
    texRef2DInput.addressMode[1] = cudaAddressModeClamp;

    // bind texture reference to array
    cudaBindTextureToArray(texRef2DInput, texInputArray);

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

    CUT_CHECK_ERROR("Pre Launch Error\n");
    // Execute Kernel Passes
    for (int i = 0; i < runs; i++) {
    int offset = i * nBlocks * blockSize;
    CudaGrayscaleDilateImageFilterKernelGlobal2D <<< nBlocks, blockSize >>> (output, imageDim, radius, kernelDim, N, offset);
    cudaThreadSynchronize();
    }
    CUT_CHECK_ERROR("Launch Error\n");

    cudaFreeArray(texInputArray);
    cudaFreeArray(texKernelArray);
    cudaUnbindTexture(texRef2DInput);
    cudaUnbindTexture(texRef2DKernel);

    }
}

#define THISTYPE float
template void CudaGrayscaleDilateImageFilterKernelFunction<THISTYPE, THISTYPE, KTYPE>(const THISTYPE * input, THISTYPE * output, 
										      const unsigned long * iDim, const unsigned long * r,
										      const KTYPE * kernel, const unsigned long * kDim,
										      const KTYPE zero, unsigned long D, unsigned long N);
#undef THISTYPE
// put these back in if we can get around float textures.

// #define THISTYPE int
// template void CudaGrayscaleDilateImageFilterKernelFunction<THISTYPE, THISTYPE, KTYPE>(const THISTYPE * input, THISTYPE * output, 
// 										     const unsigned long * iDim, const unsigned long * r,
// 										     const KTYPE * kernel, const unsigned long * kDim,
// 										     const KTYPE zero, unsigned long D, unsigned long N);
// #undef THISTYPE

// #define THISTYPE short
// template void CudaGrayscaleDilateImageFilterKernelFunction<THISTYPE, THISTYPE, KTYPE>(const THISTYPE * input, THISTYPE * output, 
// 										     const unsigned long * iDim, const unsigned long * r,
// 										     const KTYPE * kernel, const unsigned long * kDim,
// 										     const KTYPE zero, unsigned long D, unsigned long N);
// #undef THISTYPE

// #define THISTYPE unsigned char
// template void CudaGrayscaleDilateImageFilterKernelFunction<THISTYPE, THISTYPE, KTYPE>(const THISTYPE * input, THISTYPE * output, 
// 										     const unsigned long * iDim, const unsigned long * r,
// 										     const KTYPE * kernel, const unsigned long * kDim,
// 										     const KTYPE zero, unsigned long D, unsigned long N);
// #undef THISTYPE
#undef KTYPE

