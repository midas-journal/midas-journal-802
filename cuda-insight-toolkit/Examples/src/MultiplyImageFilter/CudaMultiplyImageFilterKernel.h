/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaMultiplyImageFilterKernel.h
  Language:  C++

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

template <class T, class S> extern
void MultiplyImageKernelFunction(const T* input1, const T* input2, S* output, unsigned int N);

