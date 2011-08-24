/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaFilterKernel.cu
  Language:  CUDA

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
/** \class CudaFilterKernel.cu
 * \brief Cuda kernel code
 * \author Phillip Ward, Luke Parkinson, Daniel Micevski, Christopher
 * Share, Victorian Partnership for Advanced Computing (VPAC). 
 * Richard Beare, Monash University
 */

#include <stdio.h>
#include <cuda.h>
#include <cutil.h>

template <class T>
__global__ void cuKernel(T *output, int N)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx<N) 
   {
      // Do Something Here
   }
}

float * cuFunction(const float* input, unsigned int N)
{
   float *output;

   // Method #1 - Re-use Device Memory for Output
   //
   // Cast input to non const.
   // Note: ContainerManageDevice must be set to false in input container.
   // eg: 
   /*
      output = const_cast<float*>(input);
   */

   // Method #2 - Allocate New Memory for Output
   //
   // CudaMalloc new output memory
   // Note: ContainerManageDevice must be set to true in input container.
   // 
   // eg: 
   /* 
      cudaMalloc((void **) &output, sizeof(float)*N);
   */


   // Compute execution configuration 
   int blockSize = 64;
   int nBlocks = N/blockSize + (N%blockSize == 0?0:1);


   // Call kernel
   cuKernel <<< nBlocks, blockSize >>> (output, N);

   // Return pointer to the output
   return output;
}
