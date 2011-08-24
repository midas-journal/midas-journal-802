/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaStatisticsImageFilterKernel.h
  Language:  C++

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/


template <class T> extern
void StatisticsImageKernelFunction(const T* input, 
				   T &Minimum, T &Maximum, float &Sum, float &SumOfSquares, unsigned int N);

