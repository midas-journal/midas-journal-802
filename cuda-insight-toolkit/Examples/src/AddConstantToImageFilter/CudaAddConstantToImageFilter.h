/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaAddConstantToImageFilter.h
  Language:  C++

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/


#ifndef __itkCudaAddConstantToImageFilter_h
#define __itkCudaAddConstantToImageFilter_h

#include "CudaInPlaceImageFilter.h"

namespace itk
{

/** \class CudaAddConstantToImageFilter
 *
 * \brief Add a constant to all input pixels.
 *
 * This filter is templated over the input image type
 * and the output image type.
 *
 * \author Phillip Ward, Luke Parkinson, Daniel Micevski, Christopher
 * Share, Victorian Partnership for Advanced Computing (VPAC). 
 * Richard Beare, Monash University
 *
 *
 * \ingroup IntensityImageFilters  CudaEnabled
 *
 * \sa ImageToImageFilter
 */


template <class TInputImage, class TOutputImage>
class ITK_EXPORT CudaAddConstantToImageFilter :
    public
CudaInPlaceImageFilter<TInputImage, TOutputImage >
{
public:

  typedef TInputImage                 InputImageType;
  typedef TOutputImage                OutputImageType;

  /** Standard class typedefs. */
  typedef CudaAddConstantToImageFilter  Self;
  typedef CudaInPlaceImageFilter<TInputImage,TOutputImage >
    Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(CudaAddConstantToImageFilter,
               CudaInPlaceImageFilter);

  typedef typename InputImageType::PixelType   InputPixelType;
  typedef typename OutputImageType::PixelType  OutputPixelType;

  typedef typename InputImageType::RegionType  InputImageRegionType;
  typedef typename OutputImageType::RegionType OutputImageRegionType;

  typedef typename InputImageType::SizeType    InputSizeType;
  typedef typename OutputImageType::SizeType    OutputSizeType;

  itkSetMacro(Constant, InputPixelType);
  itkGetConstReferenceMacro(Constant, InputPixelType);

  InputPixelType getConstant() const
  {
    return m_Constant;
  }

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(InputConvertibleToOutputCheck,
                  (Concept::Convertible<typename TInputImage::PixelType,
                   typename TOutputImage::PixelType>));
  itkConceptMacro(Input1Input2OutputAddOperatorCheck,
                  (Concept::AdditiveOperators<typename TInputImage::PixelType,
                   InputPixelType,
                   typename TOutputImage::PixelType>));
  /** End concept checking */
#endif

protected:
  CudaAddConstantToImageFilter();
  ~CudaAddConstantToImageFilter() {}
  void PrintSelf(std::ostream& os, Indent indent) const;
  void GenerateData();

private:
  CudaAddConstantToImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  InputPixelType m_Constant;

};

} // end namespace itk
#ifndef ITK_MANUAL_INSTANTIATION
#include "CudaAddConstantToImageFilter.txx"
#endif

#endif
