/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaFilter.txx
  Language:  

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __CudaFilter_txx
#define __CudaFilter_txx

#include "CudaFilter.h"

#include "CudaFilterKernel.h"

namespace itk
{

   /*
    *
    */
   template<class TInputImage, class TOutputImage>
      CudaFilter<TInputImage, TOutputImage>::CudaFilter()
      {
      }


   /*
    *
    */
   template <class TInputImage, class TOutputImage>
      void CudaFilter<TInputImage, TOutputImage>
      ::PrintSelf(std::ostream& os, Indent indent) const
      {
         Superclass::PrintSelf(os, indent);

         os << indent << "Cuda Filter" << std::endl;
      }

   /*
    *
    */
   template <class TInputImage, class TOutputImage>
      void CudaFilter<TInputImage, TOutputImage>
      ::GenerateData()
      {
         // Set input and output type names.
         typename OutputImageType::Pointer output = this->GetOutput();
         typename InputImageType::ConstPointer input = this->GetInput();

         // Allocate Output Region
         // This code will set the output image to the same size as the input image.
         typename OutputImageType::RegionType outputRegion;
         outputRegion.SetSize(input->GetLargestPossibleRegion().GetSize());
         outputRegion.SetIndex(input->GetLargestPossibleRegion().GetIndex());
         output->SetRegions(outputRegion);
         output->Allocate();

         // Get Total Size
         const unsigned long N = input->GetPixelContainer()->Size();

         // Pointer for output array of output pixel type
         typename TOutputImage::PixelType * ptr;

         // Call Cu Function to execute kernel
         // Return pointer is to output array
         ptr = cuFunction(input->GetDevicePointer(), N);

         // Set output array to output image
         output->GetPixelContainer()->SetDevicePointer(ptr, N, true);

         /* 
	    If Method #1 is used in the cuFunction, you must release control of the input device pointer.
            If you do not set this line, the input image will cudaFree the memory you have passed to your output image.
            If Method #2 is used, comment out the next two lines.
         */
         TInputImage * inputPtr = const_cast<TInputImage*>(this->GetInput());
         inputPtr->GetPixelContainer()->SetContainerManageDevice(false);
      }
}


#endif



