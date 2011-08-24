/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itk-cpu-binarythreshold.cxx
  Language:  C++

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
/** \class itk-cpu-binarythreshold.cxx
 * \brief Cuda kernel code
 * \author Phillip Ward, Luke Parkinson, Daniel Micevski, Christopher
 * Share, Victorian Partnership for Advanced Computing (VPAC). 
 * Richard Beare, Monash University
 */

#include <stdio.h>
#include <stdlib.h>
#include "itkImage.h"
#include "CudaTest.h"

#include "itkBinaryThresholdImageFilter.h"

using namespace std;

int main(int argc, char **argv)
{

  // Pixel Types
  typedef float InputPixelType;
  typedef float OutputPixelType;
  const unsigned int Dimension = 2;
  bool InPlace = (bool)atoi(argv[3]);

  // IO Types
  // typedef itk::RGBPixel< InputPixelType >       PixelType;
  typedef itk::Image<InputPixelType, Dimension> InputImageType;
  typedef itk::Image<OutputPixelType, Dimension> OutputImageType;
  typedef itk::ImageFileReader<InputImageType> ReaderType;
  typedef itk::ImageFileWriter<OutputImageType> WriterType;

  typedef itk::BinaryThresholdImageFilter<InputImageType, OutputImageType> FilterType;
  FilterType::Pointer filter = FilterType::New();
  filter->SetUpperThreshold(atof(argv[4]));
  filter->SetInsideValue(100);

  return(CudaTest1a<FilterType, InputImageType, OutputImageType>(InPlace, argv[1], argv[2], filter));
}

