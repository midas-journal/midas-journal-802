/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaAddConstantToImageFilterKernel.cu
  Language:  CUDA

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
/** \class CudaAddConstantToImageFilterKernel.cu
 * \brief Cuda kernel code
 * \author Phillip Ward, Luke Parkinson, Daniel Micevski, Christopher
 * Share, Victorian Partnership for Advanced Computing (VPAC). 
 * Richard Beare, Monash University
 */


#include <stdio.h>
#include <cuda.h>
#include <cutil.h>

#ifndef CITK_USE_THRUST

template <class S>
__global__ void AddConstantToImageKernel(S *output, int N, S C)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx<N) 
   {
      output[idx] += C;
   }
}

template <class T, class S>
__global__ void AddConstantToImageKernel(S *output, const T *input, int N, T C)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx<N) 
   {
      output[idx] = input[idx] + C;
   }
}

template<class T, class S>
void AddConstantToImageKernelFunction(const T* input, S* output, unsigned int N, T C)
{
   // Compute execution configuration 
   int blockSize = 128;
   int nBlocks = N/blockSize + (N%blockSize == 0?0:1);

   // Call kernel
   if (output == input)
     AddConstantToImageKernel <<< nBlocks, blockSize >>> (output, N, C);
   else
     AddConstantToImageKernel <<< nBlocks, blockSize >>> (output, input, N, C);

}

#else
#include "thrust/transform.h"

template <typename S>
struct addC
{
  const S a;

  addC(S _a) : a(_a) {}

  __host__ __device__
  S operator()(const float& x) const { 
    return a + x;
  }
};

template<class T, class S>
void AddConstantToImageKernelFunction(const T* input, S* output, unsigned int N, T C)
{
  thrust::device_ptr<const T> i1(input);
  thrust::device_ptr<S> o1(output);
  thrust::transform(i1, i1 + N, o1, addC<S>(C));

}

#endif
// versions we wish to compile
#define THISTYPE float
template void AddConstantToImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE * output, unsigned int N, THISTYPE C);
#undef THISTYPE
#define THISTYPE int
template void AddConstantToImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE *output, unsigned int N, THISTYPE C);
#undef THISTYPE

#define THISTYPE short
template void AddConstantToImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE *output, unsigned int N, THISTYPE C);
#undef THISTYPE

#define THISTYPE unsigned char
template void AddConstantToImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE *output, unsigned int N, THISTYPE C);
#undef THISTYPE

