/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaRescaleIntensityImageFilter.h
  Language:  C++

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/


#ifndef __itkCudaRescaleIntensityImageFilter_h
#define __itkCudaRescaleIntensityImageFilter_h

#include "CudaInPlaceImageFilter.h"

namespace itk
{

/** \class CudaRescaleIntensityImageFilter
 * \brief Applies a linear transformation to the intensity levels of the
 * input Image.
 *
 * RescaleIntensityImageFilter applies pixel-wise a linear transformation
 * to the intensity values of input image pixels. The linear transformation
 * is defined by the user in terms of the minimum and maximum values that
 * the output image should have.
 *
 * All computations are performed in the precison of the input pixel's
 * RealType. Before assigning the computed value to the output pixel.
 *
 * NOTE: In this filter the minimum and maximum values of the input image are
 * computed internally using the MinimumMaximumImageCalculator. Users are not
 * supposed to set those values in this filter. If you need a filter where you
 * can set the minimum and maximum values of the input, please use the
 * IntensityWindowingImageFilter. If you want a filter that can use a
 * user-defined linear transformation for the intensity, then please use the
 * ShiftScaleImageFilter.
 *
 * \author Phillip Ward, Luke Parkinson, Daniel Micevski, Christopher
 * Share, Victorian Partnership for Advanced Computing (VPAC). 
 * Richard Beare, Monash University
 *
 * \sa CudaInPlaceImageFilter
 *
 * \ingroup IntensityImageFilters  CudaEnabled
 *
 */


template <class TInputImage, class TOutputImage>
class ITK_EXPORT CudaRescaleIntensityImageFilter :
    public
CudaInPlaceImageFilter<TInputImage, TOutputImage >
{
public:

  typedef TInputImage                 InputImageType;
  typedef TOutputImage                OutputImageType;

  /** Standard class typedefs. */
  typedef CudaRescaleIntensityImageFilter  Self;
  typedef CudaInPlaceImageFilter<TInputImage,TOutputImage >
    Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(CudaRescaleIntensityImageFilter,
               CudaInPlaceImageFilter);

  typedef typename InputImageType::PixelType   InputPixelType;
  typedef typename OutputImageType::PixelType  OutputPixelType;

  typedef typename InputImageType::RegionType  InputImageRegionType;
  typedef typename OutputImageType::RegionType OutputImageRegionType;

  typedef typename InputImageType::SizeType    InputSizeType;

  typedef typename NumericTraits<InputPixelType>::RealType RealType;

  itkSetMacro(OutputMaximum, OutputPixelType);
  itkSetMacro(OutputMinimum, OutputPixelType);

  itkGetConstReferenceMacro(OutputMaximum, OutputPixelType);
  itkGetConstReferenceMacro(OutputMinimum, OutputPixelType);

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(InputHasNumericTraitsCheck,
                  (Concept::HasNumericTraits<InputPixelType>));
  itkConceptMacro(OutputHasNumericTraitsCheck,
                  (Concept::HasNumericTraits<OutputPixelType>));
  itkConceptMacro(RealTypeMultiplyOperatorCheck,
                  (Concept::MultiplyOperator<RealType>));
  itkConceptMacro(RealTypeAdditiveOperatorsCheck,
                  (Concept::AdditiveOperators<RealType>));
  /** End concept checking */
#endif

protected:
  CudaRescaleIntensityImageFilter();
  ~CudaRescaleIntensityImageFilter() {}
  void PrintSelf(std::ostream& os, Indent indent) const;
  void GenerateData();

private:
  CudaRescaleIntensityImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  OutputPixelType m_OutputMaximum;
  OutputPixelType m_OutputMinimum;

};

} // end namespace itk
#ifndef ITK_MANUAL_INSTANTIATION
#include "CudaRescaleIntensityImageFilter.txx"
#endif

#endif
