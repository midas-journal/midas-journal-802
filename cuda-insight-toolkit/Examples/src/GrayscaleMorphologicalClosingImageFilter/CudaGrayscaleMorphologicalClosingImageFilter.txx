/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaGrayscaleMorphologicalClosingImageFilter.txx
  Language:  

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __CudaGrayscaleMorphologicalClosingImageFilter_txx
#define __CudaGrayscaleMorphologicalClosingImageFilter_txx

#include "CudaGrayscaleMorphologicalClosingImageFilter.h"

namespace itk
{

/*
    *
    */
template<class TInputImage, class TOutputImage, class TKernel>
CudaGrayscaleMorphologicalClosingImageFilter<TInputImage, TOutputImage, TKernel>::CudaGrayscaleMorphologicalClosingImageFilter()
{
}


/*
    *
    */
template <class TInputImage, class TOutputImage, class TKernel>
void CudaGrayscaleMorphologicalClosingImageFilter<TInputImage, TOutputImage, TKernel>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "Cuda Filter" << std::endl;
}

/*
    *
    */
template <class TInputImage, class TOutputImage, class TKernel>
void CudaGrayscaleMorphologicalClosingImageFilter<TInputImage, TOutputImage, TKernel>
::GenerateData()
{
  // no safe boundaries here
  this->AllocateOutputs();
  // Set input and output type names.
  typename OutputImageType::Pointer output = this->GetOutput();
  typename InputImageType::ConstPointer input = this->GetInput();


  /** set up erosion and dilation methods */
  typename CudaGrayscaleDilateImageFilter<TInputImage, TOutputImage, TKernel>::Pointer
    dilate = CudaGrayscaleDilateImageFilter<TInputImage, TOutputImage, TKernel>::New();

  typename CudaGrayscaleErodeImageFilter<TOutputImage, TOutputImage, TKernel>::Pointer
    erode = CudaGrayscaleErodeImageFilter<TOutputImage, TOutputImage, TKernel>::New();

  dilate->SetKernel( this->GetKernel() );
  erode->SetKernel( this->GetKernel() );

  dilate->SetInput( input );
  erode->SetInput(  dilate->GetOutput() );

  ProgressAccumulator::Pointer progress = ProgressAccumulator::New();
  progress->SetMiniPipelineFilter(this);
  progress->RegisterInternalFilter(erode, .5f);
  progress->RegisterInternalFilter(dilate, .5f);

  dilate->SetInput( this->GetInput() );

  /** execute the minipipeline */
  erode->GraftOutput(this->GetOutput());
  erode->Update();
  this->GraftOutput(erode->GetOutput());

}
}


#endif



