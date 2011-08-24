/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaMeanImageFilter.txx
  Language:  

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __CudaMeanImageFilter_txx
#define __CudaMeanImageFilter_txx

#include "CudaMeanImageFilter.h"

#include "CudaMeanImageFilterKernel.h"

namespace itk {

/*
 *
 */
template<class TInputImage, class TOutputImage>
CudaMeanImageFilter<TInputImage, TOutputImage>::CudaMeanImageFilter() 
{
}

/*
 *
 */
template<class TInputImage, class TOutputImage>
void CudaMeanImageFilter<TInputImage, TOutputImage>::PrintSelf(
  std::ostream& os, Indent indent) const 
{
  Superclass::PrintSelf(os, indent);

  os << indent << "Cuda Filter" << std::endl;
}

/*
 *
 */
template<class TInputImage, class TOutputImage>
void CudaMeanImageFilter<TInputImage, TOutputImage>::GenerateData() 
{
  this->AllocateOutputs();
  // Set input and output type names.
  typename OutputImageType::Pointer output = this->GetOutput();
  typename InputImageType::ConstPointer input = this->GetInput();
	
  // Calculate number of Dimensions
  unsigned int D = input->GetLargestPossibleRegion().GetImageDimension();
  unsigned int N = 1;

  unsigned int imageDimX = 1;
  unsigned int imageDimY = 1;
  unsigned int imageDimZ = 1;

  unsigned int radiusX = 0;
  unsigned int radiusY = 0;
  unsigned int radiusZ = 0;

  if (D > 0) 
    {
    imageDimX = input->GetLargestPossibleRegion().GetSize()[0];
    N *= imageDimX;
    radiusX = m_Radius[0];
  }
  if (D > 1) 
    {
    imageDimY = input->GetLargestPossibleRegion().GetSize()[1];
    N *= imageDimY;
    radiusY = m_Radius[1];
    }
  if (D > 2) 
    {
    imageDimZ = input->GetLargestPossibleRegion().GetSize()[2];
    N *= imageDimZ;
    radiusZ = m_Radius[2];
    }
  if (D > 3) 
    {
    std::cerr << "Only up to 3 dimensions supported" << std::endl;
    }

  // Call Cu Function to execute kernel
  // Return pointer is to output array
  CudaMeanImageFilterKernelFunction(input->GetDevicePointer(),
				    output->GetDevicePointer(),
				    imageDimX, imageDimY, imageDimZ, radiusX, radiusY, radiusZ, N);

}
}

#endif

