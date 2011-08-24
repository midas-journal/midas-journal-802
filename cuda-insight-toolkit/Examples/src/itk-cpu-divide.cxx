/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itk-cpu-divide.cxx
  Language:  C++

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
/** \class itk-cpu-divide.cxx
 * \brief Cuda kernel code
 * \author Phillip Ward, Luke Parkinson, Daniel Micevski, Christopher
 * Share, Victorian Partnership for Advanced Computing (VPAC). 
 * Richard Beare, Monash University
 */

#include <stdio.h>
#include <stdlib.h>

#include "itkImage.h"
#include "itkDivideImageFilter.h"

using namespace std;

#include "CudaTest.h"

int main(int argc, char **argv) {
  int nFilters = atoi(argv[3]);
  bool InPlace = (bool)atoi(argv[4]);
  const unsigned Dimension = 2;
  typedef float InputPixelType;
  typedef float OutputPixelType;

  typedef itk::Image<InputPixelType, Dimension> InputImageType;
  typedef itk::Image<OutputPixelType, Dimension> OutputImageType;
  typedef itk::DivideImageFilter<InputImageType, InputImageType, OutputImageType> FilterType;

  return(CudaTest2<FilterType, InputImageType, OutputImageType>(nFilters, InPlace, argv[1], argv[1], argv[2]));
}
