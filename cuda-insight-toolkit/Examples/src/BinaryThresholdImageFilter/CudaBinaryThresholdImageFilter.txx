/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaBinaryThresholdImageFilter.txx
  Language:  

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __CudaBinaryThresholdImageFilter_txx
#define __CudaBinaryThresholdImageFilter_txx

#include "CudaBinaryThresholdImageFilter.h"

#include "CudaBinaryThresholdImageFilterKernel.h"

#include <vector>
#include <algorithm>

namespace itk
{

/*
    *
    */
template<class TInputImage, class TOutputImage>
CudaBinaryThresholdImageFilter<TInputImage, TOutputImage>::CudaBinaryThresholdImageFilter()
{
   m_InsideValue = NumericTraits<OutputPixelType>::max();
   m_OutsideValue = NumericTraits<OutputPixelType>::Zero;
   m_LowerThreshold = NumericTraits<InputPixelType>::NonpositiveMin();
   m_UpperThreshold = NumericTraits<InputPixelType>::max();

//    m_InsideValue = 1;
//    m_OutsideValue = 0;
//    m_LowerThreshold = 0;
//    m_UpperThreshold = 100;
}


/*
    *
    */
template <class TInputImage, class TOutputImage>
void CudaBinaryThresholdImageFilter<TInputImage, TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "Cuda Binary Threshold Filter" << std::endl;
}

/*
    *
    */
template <class TInputImage, class TOutputImage>
void CudaBinaryThresholdImageFilter<TInputImage, TOutputImage>
::GenerateData()
{
  this->AllocateOutputs();
  typename OutputImageType::Pointer output = this->GetOutput();
  typename InputImageType::ConstPointer input = this->GetInput();

	
  // Get Total Size
  const unsigned long N = input->GetPixelContainer()->Size();

  // Call Cuda Function
  BinaryThresholdImageKernelFunction<InputPixelType, OutputPixelType>(input->GetDevicePointer(), 
				     output->GetDevicePointer(),
				     m_LowerThreshold, m_UpperThreshold, m_InsideValue,
				     m_OutsideValue, N);
}
}


#endif



