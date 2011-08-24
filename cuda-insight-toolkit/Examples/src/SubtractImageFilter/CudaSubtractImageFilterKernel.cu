/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaSubtractImageFilterKernel.cu
  Language:  CUDA

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
/** \class CudaSubtractImageFilterKernel.cu
 * \brief Cuda kernel code
 * \author Phillip Ward, Luke Parkinson, Daniel Micevski, Christopher
 * Share, Victorian Partnership for Advanced Computing (VPAC). 
 * Richard Beare, Monash University
 */

#include <stdio.h>
#include <cuda.h>
#include <cutil.h>

template <class T, class S>
__global__ void subtractImage(S *output, const T *input, int N)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx<N) 
   {
      output[idx] -= input[idx];
   }
}

template <class T, class S>
__global__ void subtractImage(S *output, const T *input1, const T *input2, int N)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx<N) 
   {
      output[idx] = input1[idx] - input2[idx];
   }
}

template <class T, class S>
void SubtractImageKernelFunction(const T* input1, const T* input2, S *output, unsigned int N)
{
  // Compute execution configuration 
   int blockSize = 128;
   int nBlocks = N/blockSize + (N%blockSize == 0?0:1);

   // Call kernel
   if (output == input1)
     subtractImage <<< nBlocks, blockSize >>> (output, input2, N);
   else
     subtractImage <<< nBlocks, blockSize >>> (output, input1, input2, N);

}

// versions we wish to compile
#define THISTYPE float
template void SubtractImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, const THISTYPE * input2, THISTYPE * output, unsigned int N);
#undef THISTYPE
#define THISTYPE int
template void SubtractImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, const THISTYPE * input2, THISTYPE *output, unsigned int N);
#undef THISTYPE

#define THISTYPE short
template void SubtractImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, const THISTYPE * input2, THISTYPE *output, unsigned int N);
#undef THISTYPE

#define THISTYPE char
template void SubtractImageKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input1, const THISTYPE * input2,  THISTYPE *output, unsigned int N);
#undef THISTYPE

