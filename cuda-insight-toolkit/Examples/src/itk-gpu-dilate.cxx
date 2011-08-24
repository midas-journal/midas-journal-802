/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itk-gpu-dilate.cxx
  Language:  C++

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
/** \class itk-gpu-dilate.cxx
 * \brief Cuda kernel code
 * \author Phillip Ward, Luke Parkinson, Daniel Micevski, Christopher
 * Share, Victorian Partnership for Advanced Computing (VPAC). 
 * Richard Beare, Monash University
 */

#include <stdio.h>
#include <stdlib.h>

#include "itkImage.h"
#include "CudaTest.h"
#include "itkBinaryBallStructuringElement.h"
#include "CudaGrayscaleDilateImageFilter.h"

using namespace std;

#include <stdio.h>
#include <stdlib.h>

using namespace std;

int main(int argc, char **argv) {

  // Pixel Types
  typedef float InputPixelType;
  typedef float OutputPixelType;
  const unsigned int Dimension = 2;
  int rad = atoi(argv[3]);

  // IO Types
  // typedef itk::RGBPixel< InputPixelType >       PixelType;
  typedef itk::Image<InputPixelType, Dimension> InputImageType;
  typedef itk::Image<OutputPixelType, Dimension> OutputImageType;
  // structuring element
  typedef itk::BinaryBallStructuringElement<unsigned char, Dimension> StructuringElementType;

  typedef itk::CudaGrayscaleDilateImageFilter<InputImageType, OutputImageType, StructuringElementType> FilterType;
  StructuringElementType  structuringElement;
  InputImageType::SizeType radius;
  radius.Fill(rad);
  
  structuringElement.SetRadius( rad );
  structuringElement.CreateStructuringElement();
  
  FilterType::Pointer filter = FilterType::New();
  filter->SetKernel(structuringElement);
  return(CudaTest1b<FilterType, InputImageType, OutputImageType>(argv[1], argv[2], filter));
}

