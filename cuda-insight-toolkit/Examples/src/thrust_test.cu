/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    thrust_test.cu
  Language:  CUDA

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
/** \class thrust_test.cu
 * \brief Cuda kernel code
 * \author Phillip Ward, Luke Parkinson, Daniel Micevski, Christopher
 * Share, Victorian Partnership for Advanced Computing (VPAC). 
 * Richard Beare, Monash University
 */

// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
// #include <thrust/generate.h>
// #include <thrust/reduce.h>
// #include <thrust/functional.h>
// #include <cstdlib>

int main(void)
{
  int *dInt;
  cudaMalloc(&dInt, sizeof(int)*1);

  // generate random data on the host
  //thrust::host_vector<int> h_vec(100);
  //thrust::generate(h_vec.begin(), h_vec.end(), rand);

  // transfer to device and compute sum
  //thrust::device_vector<int> d_vec(100);
  //int x = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());
  return 0;
}
