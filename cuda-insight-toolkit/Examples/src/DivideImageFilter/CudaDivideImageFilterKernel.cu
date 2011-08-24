/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaDivideImageFilterKernel.cu
  Language:  CUDA

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
/** \class CudaDivideImageFilterKernel.cu
 * \brief Cuda kernel code
 * \author Phillip Ward, Luke Parkinson, Daniel Micevski, Christopher
 * Share, Victorian Partnership for Advanced Computing (VPAC). 
 * Richard Beare, Monash University
 */

#include <stdio.h>
#include <cuda.h>
#include <cutil.h>

template <class T, class S>
__global__ void DivideImageKernel(S *output, const T *input, int N, S MAX)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx<N) 
   {
      output[idx] = output[idx] / input[idx];
      if (input[idx]==0) { output[idx] = MAX; }
   }
}

template <class T, class S>
__global__ void DivideImageKernel(S *output, const T *input1, const T *input2, int N, S MAX)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx<N) 
   {
      output[idx] = output[idx] / input1[idx];
      if (input1[idx]==0) { output[idx] = MAX; }
   }
}

template <class T, class S>
void DivideImageKernelFunction(const T* input1, const T* input2, S *output, unsigned int N, S MAX)
{
   // Compute execution configuration 
   int blockSize = 64;
   int nBlocks = N/blockSize + (N%blockSize == 0?0:1);


   // Call kernel
   if (output == input1)
     DivideImageKernel <<< nBlocks, blockSize >>> (output, input2, N, MAX);
   else
     DivideImageKernel <<< nBlocks, blockSize >>> (output, input1, input2, N, MAX);
}

// versions we wish to compile
#define THISTYPE float
template void DivideImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, const THISTYPE * input2, THISTYPE * output, unsigned int N, THISTYPE MAX);
#undef THISTYPE
#define THISTYPE int
template void DivideImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, const THISTYPE * input2, THISTYPE *output, unsigned int N, THISTYPE MAX);
#undef THISTYPE

#define THISTYPE short
template void DivideImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, const THISTYPE * input2, THISTYPE *output, unsigned int N, THISTYPE MAX);
#undef THISTYPE

#define THISTYPE unsigned char
template void DivideImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, const THISTYPE * input2,  THISTYPE *output, unsigned int N, THISTYPE MAX);
#undef THISTYPE

