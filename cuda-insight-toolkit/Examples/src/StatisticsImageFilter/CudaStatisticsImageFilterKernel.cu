/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaStatisticsImageFilterKernel.cu
  Language:  CUDA

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
/** \class CudaStatisticsImageFilterKernel.cu
 * \brief Cuda kernel code
 * \author Phillip Ward, Luke Parkinson, Daniel Micevski, Christopher
 * Share, Victorian Partnership for Advanced Computing (VPAC). 
 * Richard Beare, Monash University
 */

#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <cutil.h>
#include "CudaStatisticsImageFilterKernel.h"
#include <limits>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>


template <typename T>
struct stats_square
{
  __host__ __device__
  float operator()(const T& x) const { 
    return (float)x * (float)x;
  }
};

template <typename T>
struct stats_cast
{
  __host__ __device__
  float operator()(const T& x) const { 
    return (float)x;
  }
};

template <class T> 
void StatisticsImageKernelFunction(const T* input, 
				   T &Minimum, T &Maximum, float &Sum, 
				   float &SumOfSquares, unsigned int N) 
{
  thrust::device_ptr<const T> dptr(input);
  Maximum = thrust::reduce(dptr, dptr+N, std::numeric_limits<T>::min(), thrust::maximum<T>());

  Minimum = thrust::reduce(dptr, dptr+N, std::numeric_limits<T>::max(), thrust::minimum<T>());

  // using transform_reduce to include casting
  Sum = thrust::transform_reduce(dptr, dptr+N, stats_cast<T>(), 0, thrust::plus<float>());

  SumOfSquares = thrust::transform_reduce(dptr, dptr + N, stats_square<T>(), 0, thrust::plus<float>());
}
// versions we wish to compile
#define THISTYPE float
template void StatisticsImageKernelFunction<THISTYPE>(const THISTYPE * input, THISTYPE &Minimum, THISTYPE &Maximum, 
						      float &Sum, float &SumOfSquares, unsigned int N);
#undef THISTYPE

#define THISTYPE int
template void StatisticsImageKernelFunction<THISTYPE>(const THISTYPE * input, THISTYPE &Minimum, THISTYPE &Maximum, 
						      float &Sum, float &SumOfSquares, unsigned int N);

#undef THISTYPE

#define THISTYPE short
template void StatisticsImageKernelFunction<THISTYPE>(const THISTYPE * input, THISTYPE &Minimum, THISTYPE &Maximum, 
						      float &Sum, float &SumOfSquares, unsigned int N);

#undef THISTYPE

#define THISTYPE unsigned char
template void StatisticsImageKernelFunction<THISTYPE>(const THISTYPE * input, THISTYPE &Minimum, THISTYPE &Maximum, 
						      float &Sum, float &SumOfSquares, unsigned int N);

#undef THISTYPE
