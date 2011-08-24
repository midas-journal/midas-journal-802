/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaSubtractConstantFromImageFilter.txx
  Language:  

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __CudaSubtractConstantFromImageFilter_txx
#define __CudaSubtractConstantFromImageFilter_txx

#include "CudaSubtractConstantFromImageFilter.h"

#include "CudaSubtractConstantFromImageFilterKernel.h"

namespace itk {

/*
 *
 */
template<class TInputImage, class TOutputImage>
CudaSubtractConstantFromImageFilter<TInputImage, TOutputImage>::CudaSubtractConstantFromImageFilter()
{
  m_Constant = 0;
}

/*
 *
 */
template<class TInputImage, class TOutputImage>
void CudaSubtractConstantFromImageFilter<TInputImage, TOutputImage>::PrintSelf(
  std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "Cuda SubtractConstantFrom Image Filter" << std::endl;
}

/*
 *
 */
template<class TInputImage, class TOutputImage>
void CudaSubtractConstantFromImageFilter<TInputImage, TOutputImage>::GenerateData() {
  // Set input and output type names.
  this->AllocateOutputs();
  typename OutputImageType::Pointer output = this->GetOutput();
  typename InputImageType::ConstPointer input = this->GetInput();


  // Get Total Size
  const unsigned long N = input->GetPixelContainer()->Size();
  // Call Cu Function to execute kernel
  // Return pointer is to output array
  SubtractConstantFromImageKernelFunction<InputPixelType, OutputPixelType>(input->GetDevicePointer(), output->GetDevicePointer(),
									   N, m_Constant);

}
}

#endif

