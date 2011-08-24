/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaAbsImageFilterKernel.cu
  Language:  CUDA

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
/** \class CudaAbsImageFilterKernel.cu
 * \brief Cuda kernel code
 * \author Phillip Ward, Luke Parkinson, Daniel Micevski, Christopher
 * Share, Victorian Partnership for Advanced Computing (VPAC). 
 * Richard Beare, Monash University
 */

#include <stdio.h>
#include <cuda.h>

#ifndef CITK_USE_THRUST
template <class T>
__global__ void AbsImageKernel(T *output, int N)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;

   if (idx<N) 
   {
	   T temp = output[idx];
      output[idx] = (temp < 0) ? -temp : temp;
   }
}

template <class T, class S>
__global__ void AbsImageKernel(S *output, const T *input, int N)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;

   if (idx<N) 
   {
   T temp = input[idx];
   output[idx] = (temp < 0) ? -temp : temp;
   }
}

template <class T, class S>
void AbsImageKernelFunction(const T * input, S * output, unsigned int N)
{
   // Compute execution configuration 
   int blockSize = 128;
   int nBlocks = N/blockSize + (N%blockSize == 0?0:1);

   // Call kernel
   if (input == output) 
     {
     AbsImageKernel<<< nBlocks, blockSize >>> (output, N);
     }
   else
     {
     AbsImageKernel<<< nBlocks, blockSize >>> (output, input, N);
     }

}

#else
#include "thrust/transform.h"
#include "thrust/functional.h"
template <typename T> 
struct ABS 
{ 
  __host__ __device__ 
  T operator()(const T& x) const { 
    return abs(x); 
  } 
}; 

template <class T, typename S>
void AbsImageKernelFunction(const T * input, S * output, unsigned int N)
{
  thrust::device_ptr<const T> i1(input);
  thrust::device_ptr<S> o1(output);
  // absolute_value is deprecated in thrust - not sure what to replace
  // it with
  thrust::transform(i1, i1 + N, o1, ABS<S>());
}

#endif

// versions we wish to compile
#define THISFUNC AbsImageKernelFunction
#define THISTYPE float
template void THISFUNC<THISTYPE, THISTYPE>(const THISTYPE * input,   THISTYPE * output, unsigned int N);
#undef THISTYPE
#define THISTYPE int
template void  THISFUNC<THISTYPE, THISTYPE>(const THISTYPE * input,  THISTYPE * output, unsigned int N);
#undef THISTYPE

#define THISTYPE short
template void  THISFUNC<THISTYPE, THISTYPE>(const THISTYPE * input,  THISTYPE * output, unsigned int N);
#undef THISTYPE

#undef THISFUNC
