/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaSubtractConstantFromImageFilterKernel.h
  Language:  C++

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
/*
 * File Name:    CudaSubtractConstantFromImageFilterKernel.h
 *
 * Author:        Phillip Ward, Richard Beare
 * Creation Date: Monday, January 18 2010, 10:00
 * Last Modified: Fri May  6 15:32:36 EST 2011
 *
 * File Description:
 *
 */
template <class T, class S> extern
void SubtractConstantFromImageKernelFunction(const T* input1, S*output, unsigned int N, T C);

