/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itk-gpu-rescaleintensity.cxx
  Language:  C++

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
/** \class itk-gpu-rescaleintensity.cxx
 * \brief Cuda kernel code
 * \author Phillip Ward, Luke Parkinson, Daniel Micevski, Christopher
 * Share, Victorian Partnership for Advanced Computing (VPAC). 
 * Richard Beare, Monash University
 */

#include <stdio.h>
#include <stdlib.h>

#include "itkImage.h"
#include "CudaRescaleIntensityImageFilter.h"
#include "CudaTest.h"

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

  typedef itk::CudaRescaleIntensityImageFilter<InputImageType, OutputImageType> FilterType;
  FilterType::Pointer filter = FilterType::New();
  filter->SetOutputMinimum(atof(argv[4]));
  filter->SetOutputMaximum(atof(argv[5]));
  return(CudaTest1a<FilterType, InputImageType, OutputImageType>(InPlace, argv[1], argv[2], filter));
}


