/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    external_dependency.h
  Language:  C++

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef EXTERNDEPENDENCY__H
#define EXTERNDEPENDENCY__H

#include "external_dependency3.h"
typedef unsigned int Size; 

#define CHECK_CUDA_ERROR() \
  { \
    cudaThreadSynchronize(); \
    cudaError_t error = cudaGetLastError(); \
    if(error != cudaSuccess) { \
      printf("error (%s: line %d): %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
      return 1; \
    } \
  }
#endif
