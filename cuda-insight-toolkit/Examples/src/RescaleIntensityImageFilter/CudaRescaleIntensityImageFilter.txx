/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaRescaleIntensityImageFilter.txx
  Language:  

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __CudaRescaleIntensityImageFilter_txx
#define __CudaRescaleIntensityImageFilter_txx

#include "CudaRescaleIntensityImageFilter.h"

#include "CudaRescaleIntensityImageFilterKernel.h"

namespace itk
{

/*
    *
    */
template<class TInputImage, class TOutputImage>
CudaRescaleIntensityImageFilter<TInputImage, TOutputImage>::CudaRescaleIntensityImageFilter()
{
  m_OutputMaximum = NumericTraits<typename TOutputImage::PixelType>::max();
  m_OutputMinimum = NumericTraits<typename TOutputImage::PixelType>::NonpositiveMin();
}


/*
    *
    */
template <class TInputImage, class TOutputImage>
void CudaRescaleIntensityImageFilter<TInputImage, TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "Cuda Filter" << std::endl;
}

/*
    *
    */
template <class TInputImage, class TOutputImage>
void CudaRescaleIntensityImageFilter<TInputImage, TOutputImage>
::GenerateData()
{
  this->AllocateOutputs();
  // Set input and output type names.
  typename OutputImageType::Pointer output = this->GetOutput();
  typename InputImageType::ConstPointer input = this->GetInput();


  // Get Total Size
  const unsigned long N = input->GetPixelContainer()->Size();


  // Call Cu Function to execute kernel
  // Return pointer is to output array
  CudaRescaleIntensityKernelFunction<InputPixelType, OutputPixelType>(input->GetDevicePointer(), 
								      output->GetDevicePointer(),
								      m_OutputMaximum, m_OutputMinimum, N);
}
}


#endif



