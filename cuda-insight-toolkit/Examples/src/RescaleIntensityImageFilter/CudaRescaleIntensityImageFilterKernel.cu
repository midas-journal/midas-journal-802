/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaRescaleIntensityImageFilterKernel.cu
  Language:  CUDA

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
/** \class CudaRescaleIntensityImageFilterKernel.cu
 * \brief Cuda kernel code
 * \author Phillip Ward, Luke Parkinson, Daniel Micevski, Christopher
 * Share, Victorian Partnership for Advanced Computing (VPAC). 
 * Richard Beare, Monash University
 */

#include <stdio.h>
#include <cuda.h>
#include <cutil.h>

#include <thrust/reduce.h>

template <class T>
__global__ void MaxMinKernel(T *maxImage, T *minImage, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float offset = N/2;   

  for ( ; ((int)offset) > 0; )   
    {
    if (idx < offset)  {   		
    maxImage[idx] = (maxImage[idx] > maxImage[(int)(idx + offset)] ? maxImage[idx] : maxImage[(int)(idx + offset)]);
    minImage[idx] = (minImage[idx] < minImage[(int)(idx + offset)] ? minImage[idx] : minImage[(int)(idx + offset)]);
    } else {
    return;
    }
    offset /= 2;
    __syncthreads();
    }
}

template <class T>
__global__ void RescaleIntensityKernel(T *output, float offset, float factor, T max, T min, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
   
  if (idx < N)
    {
    T temp = output[idx] * factor + offset;
    output[idx] = (temp > max ? max : temp);
    output[idx] = (temp < min ? min : temp);
    }   
}

template <class T, class S>
__global__ void RescaleIntensityKernel(S *output, const T *input, float offset, float factor, T max, T min, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
   
  if (idx < N)
    {
    T temp = input[idx] * factor + offset;
    output[idx] = (temp > max ? max : temp);
    output[idx] = (temp < min ? min : temp);
    }   
}

template <class T, class S>
void CudaRescaleIntensityKernelFunction(const T* input, S* output, S outputMax, S outputMin, unsigned int N)
{

  // Compute execution configuration 
  int blockSize = 256;
  int nBlocks = N/(blockSize*2) + (N%(blockSize*2) == 0?0:1);

  // Call kernel
  T *tmpImage; // we will use this for the max and min calculations
  cudaMalloc(&tmpImage, sizeof(T)*N);
  cudaMemcpy(tmpImage, input, sizeof(T)*N, cudaMemcpyDeviceToDevice);


  thrust::device_ptr<T> dptr(tmpImage);

  T inputMax = thrust::reduce(dptr, dptr+N, std::numeric_limits<T>::min(), thrust::maximum<T>());
  T inputMin = thrust::reduce(dptr, dptr+N, std::numeric_limits<T>::max(), thrust::minimum<T>());

   
  cudaFree(tmpImage);
  float m_Factor = 0;
  float m_Offset = 0;
   
  if (inputMin != inputMax)
    {
    m_Factor = (outputMax-outputMin) / (inputMax-inputMin);
    }
  else if (inputMax != 0)
    {
    m_Factor = (outputMax-outputMin) / (inputMax);
    }  		
  else
    {
    m_Factor = 0;
    }
   
  m_Offset = outputMin-inputMin * m_Factor;
   
  nBlocks = N/(blockSize) + (N%blockSize == 0?0:1);

  if (input == output)
    RescaleIntensityKernel <<< nBlocks, blockSize >>> (output, m_Offset, m_Factor, outputMax, outputMin, N);
  else
    RescaleIntensityKernel <<< nBlocks, blockSize >>> (output, input, m_Offset, m_Factor, outputMax, outputMin, N);

   
}

// versions we wish to compile
#define THISTYPE float
template void CudaRescaleIntensityKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE * output, THISTYPE outputMax, THISTYPE outputMin, unsigned int N);
#undef THISTYPE

#define THISTYPE int
template void CudaRescaleIntensityKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE * output, THISTYPE outputMax, THISTYPE outputMin, unsigned int N);

#undef THISTYPE

#define THISTYPE short
template void CudaRescaleIntensityKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE * output, THISTYPE outputMax, THISTYPE outputMin, unsigned int N);

#undef THISTYPE

#define THISTYPE unsigned char
template void CudaRescaleIntensityKernelFunction<THISTYPE, THISTYPE>(const THISTYPE * input, THISTYPE * output, THISTYPE outputMax, THISTYPE outputMin, unsigned int N);

#undef THISTYPE

