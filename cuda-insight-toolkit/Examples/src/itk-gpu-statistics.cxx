/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itk-gpu-statistics.cxx
  Language:  C++

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
/** \class itk-gpu-statistics.cxx
 * \brief Cuda kernel code
 * \author Phillip Ward, Luke Parkinson, Daniel Micevski, Christopher
 * Share, Victorian Partnership for Advanced Computing (VPAC). 
 * Richard Beare, Monash University
 */

#include <stdio.h>
#include <stdlib.h>

#include "itkImage.h"
#include "CudaStatisticsImageFilter.h"
#include "itkStatisticsImageFilter.h"
#include "CudaTest.h"

using namespace std;

int main(int argc, char **argv) 
{
  const unsigned Dimension = 2;
  typedef unsigned char InputPixelType;
  typedef unsigned char OutputPixelType;

  typedef itk::Image<InputPixelType, Dimension> InputImageType;
  typedef itk::Image<OutputPixelType, Dimension> OutputImageType;
  typedef itk::CudaStatisticsImageFilter<InputImageType> FilterType;
  FilterType::Pointer filter = FilterType::New();

  typedef itk::StatisticsImageFilter<InputImageType> FilterType2;
  FilterType2::Pointer filter2 = FilterType2::New();

  int status = CudaTest1b<FilterType, InputImageType, OutputImageType>(argv[1], argv[2], filter);
  int status2 = CudaTest1b<FilterType2, InputImageType, OutputImageType>(argv[1], argv[2], filter2);
  cout << "Statistic Output" << endl;
  cout << "Minimum: " << (int)filter->GetMinimum() << endl;
  cout << "Maximum: " << (int)filter->GetMaximum() << endl;
  cout << "Mean: " << filter->GetMean() << endl;
  cout << "Sigma: " << filter->GetSigma() << endl;
  cout << "Variance: " << filter->GetVariance() << endl;
  cout << "Sum: " << filter->GetSum() << endl;


  if (filter2->GetMean() != filter->GetMean())
    {
    std::cerr << "Means are different " << filter2->GetMean()  << " " << filter->GetMean() << " " << status << " " << status2 << std::endl;
    return (EXIT_FAILURE);
    }

  return(EXIT_SUCCESS);

}
