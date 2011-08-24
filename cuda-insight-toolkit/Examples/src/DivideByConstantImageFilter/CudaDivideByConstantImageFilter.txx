/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaDivideByConstantImageFilter.txx
  Language:  

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __CudaDivideByConstantImageFilter_txx
#define __CudaDivideByConstantImageFilter_txx

#include "CudaDivideByConstantImageFilter.h"

#include "CudaDivideByConstantImageFilterKernel.h"

namespace itk
{

   /*
    *
    */
   template<class TInputImage, class TOutputImage>
      CudaDivideByConstantImageFilter<TInputImage, TOutputImage>::CudaDivideByConstantImageFilter()
      {
         m_Constant = 1;
      }


   /*
    *
    */
   template <class TInputImage, class TOutputImage>
      void CudaDivideByConstantImageFilter<TInputImage, TOutputImage>
      ::PrintSelf(std::ostream& os, Indent indent) const
      {
         Superclass::PrintSelf(os, indent);

         os << indent << "Cuda DivideByConstant Image Filter" << std::endl;
      }

   /*
    *
    */
   template <class TInputImage, class TOutputImage>
      void CudaDivideByConstantImageFilter<TInputImage, TOutputImage>
      ::GenerateData()
      {
	this->AllocateOutputs();
	// Set input and output type names.
	typename OutputImageType::Pointer output = this->GetOutput();
         typename InputImageType::ConstPointer input = this->GetInput();


         const unsigned long N = input->GetPixelContainer()->Size();

         // Call Cu Function to execute kernel
         // Return pointer is to output array
         DivideByConstantImageKernelFunction<InputPixelType, OutputPixelType>(input->GetDevicePointer(),
									      output->GetDevicePointer(), 
									      N, m_Constant);
      }
}


#endif



