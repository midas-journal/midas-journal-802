/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaNeighborhoodFilterKernel.h
  Language:  C++

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
float * CudaNeighborhoodFilterKernelFunction(const float* input, unsigned int imageDimX, unsigned int imageDimY, unsigned int imageDimZ,
		unsigned int radiusX, unsigned int radiusY, unsigned int radiusZ, unsigned int N);
