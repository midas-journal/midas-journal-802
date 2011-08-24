/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaMinimumImageFilterKernel.cu
  Language:  CUDA

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
/** \class CudaMinimumImageFilterKernel.cu
 * \brief Cuda kernel code
 * \author Phillip Ward, Luke Parkinson, Daniel Micevski, Christopher
 * Share, Victorian Partnership for Advanced Computing (VPAC). 
 * Richard Beare, Monash University
 */

#include <stdio.h>
#include <cuda.h>
#include <cutil.h>

template <class T, class S>
__global__ void MinimumImageKernel(S *output, const T *input, int N)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx<N) 
   {
           S inputValue = input[idx];
      if (output[idx] > inputValue)
      {
         output[idx] = inputValue;
      }
   }
}

template <class T, class S>
__global__ void MinimumImageKernel(S *output, const T *input1, const T* input2, int N)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx<N) 
   {
   T in1 = input1[idx];
   T in2 = input2[idx]; 
   output[idx] = (in1<in2)?in1:in2;
   }
}

template <class T, class S>
void MinimumImageKernelFunction(const T* input1, const T* input2, S* output, unsigned int N)
{
   // Compute execution configuration 
   int blockSize = 128;
   int nBlocks = N/blockSize + (N%blockSize == 0?0:1);

   // Call kernel
   if (output == input1)
     MinimumImageKernel <<< nBlocks, blockSize >>> (output, input2, N);
   else
     MinimumImageKernel <<< nBlocks, blockSize >>> (output, input1, input2, N);

   // Return pointer to the output
   // Return pointer to the output
}

// versions we wish to compile
#define THISTYPE float
template void MinimumImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, const THISTYPE * input2, THISTYPE * output, unsigned int N);
#undef THISTYPE
#define THISTYPE int
template void MinimumImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, const THISTYPE * input2, THISTYPE *output, unsigned int N);
#undef THISTYPE

#define THISTYPE short
template void MinimumImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, const THISTYPE * input2, THISTYPE *output, unsigned int N);
#undef THISTYPE

#define THISTYPE unsigned char
template void MinimumImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, const THISTYPE * input2,  THISTYPE *output, unsigned int N);
#undef THISTYPE
