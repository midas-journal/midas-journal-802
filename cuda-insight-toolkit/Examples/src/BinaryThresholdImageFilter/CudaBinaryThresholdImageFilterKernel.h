/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaBinaryThresholdImageFilterKernel.h
  Language:  C++

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

template <class T, class S> extern
void BinaryThresholdImageKernelFunction(const T* input, S* output, 
					T m_LowerThreshold,
					T m_UpperThreshold, 
					S m_InsideValue, 
					S m_OutsideValue,
					unsigned int N);

