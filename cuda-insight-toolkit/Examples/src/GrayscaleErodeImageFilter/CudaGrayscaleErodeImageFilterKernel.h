/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaGrayscaleErodeImageFilterKernel.h
  Language:  C++

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

template <class T, class S, class K> extern
void CudaGrayscaleErodeImageFilterKernelFunction(const T* input, S* output,
						 const unsigned long * imageDim, 
						 const unsigned long * radius,
						 const K * kernel, 
						 const unsigned long * kernelDim, const K zero,
						 unsigned long D, unsigned long N);
