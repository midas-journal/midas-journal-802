/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaDivideImageFilter.h
  Language:  C++

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/


#ifndef __itkCudaDivideImageFilter_h
#define __itkCudaDivideImageFilter_h

#include "CudaInPlaceImageFilter.h"

namespace itk
{

/** \class CudaDivideImageFilter
 * \brief Implements an operator for pixel-wise division of two images.
 *
 * This class is parametrized over the types of the two
 * input images and the type of the output image. When the divisor is zero,
 * the division result is set to the maximum number that can be represneted  by default to
 * avoid exception. Numeric conversions (castings) are done by the C++ defaults.
 *
 * \author Phillip Ward, Luke Parkinson, Daniel Micevski, Christopher
 * Share, Victorian Partnership for Advanced Computing (VPAC). 
 * Richard Beare, Monash University
 *
 *
 * \ingroup IntensityImageFilters  CudaEnabled
 *
 * \sa CudaInPlaceImageFilter
 */


template <class TInputImage, class TOutputImage>
class ITK_EXPORT CudaDivideImageFilter :
    public
CudaInPlaceImageFilter<TInputImage, TOutputImage >
{
public:

  typedef TInputImage                 InputImageType;
  typedef TOutputImage                OutputImageType;

  /** Standard class typedefs. */
  typedef CudaDivideImageFilter  Self;
  typedef CudaInPlaceImageFilter<TInputImage,TOutputImage >
    Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(CudaDivideImageFilter,
               CudaInPlaceImageFilter);

  typedef typename InputImageType::PixelType   InputPixelType;
  typedef typename OutputImageType::PixelType  OutputPixelType;

  typedef typename InputImageType::RegionType  InputImageRegionType;
  typedef typename OutputImageType::RegionType OutputImageRegionType;

  typedef typename InputImageType::SizeType    InputSizeType;
  typedef typename OutputImageType::SizeType    OutputSizeType;

  void SetInput1( const TInputImage * image1 )
  {
    // Process object is not const-correct
    // so the const casting is required.
    SetNthInput(0, const_cast
		<TInputImage *>( image1 ));
  }

  void SetInput2( const TInputImage * image2 )
  {
    // Process object is not const-correct
    // so the const casting is required.
    SetNthInput(1, const_cast
		<TInputImage *>( image2 ));
  }

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(IntConvertibleToInput2Check,
		  (Concept::Convertible<int, typename TInputImage::PixelType>));
  itkConceptMacro(Input1Input2OutputDivisionOperatorsCheck,
		  (Concept::DivisionOperators<typename TInputImage::PixelType,
		   typename TInputImage::PixelType,
		   typename TOutputImage::PixelType>));
  /** End concept checking */
#endif


protected:
  CudaDivideImageFilter();
  ~CudaDivideImageFilter() {}
  void PrintSelf(std::ostream& os, Indent indent) const;
  void GenerateData();

private:
  CudaDivideImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk
#ifndef ITK_MANUAL_INSTANTIATION
#include "CudaDivideImageFilter.txx"
#endif

#endif
