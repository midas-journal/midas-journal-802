/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaMeanImageFilter.h
  Language:  C++

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#ifndef __itkCudaMeanImageFilter_h
#define __itkCudaMeanImageFilter_h

#include "CudaImageToImageFilter.h"

namespace itk {

/** \class CudaMeanImageFilter
 * \brief Applies an averaging filter to an image
 *
 * Computes an image where a given pixel is the mean value of the
 * the pixels in a neighborhood about the corresponding input pixel.
 *
 * A mean filter is one of the family of linear filters.
 *
 * \author Phillip Ward, Luke Parkinson, Daniel Micevski, Christopher
 * Share, Victorian Partnership for Advanced Computing (VPAC). 
 * Richard Beare, Monash University
 *
 * \sa ImageToImageFilter
 *
 * \ingroup IntensityImageFilters  CudaEnabled
 */


template<class TInputImage, class TOutputImage>
class ITK_EXPORT CudaMeanImageFilter: public CudaImageToImageFilter<TInputImage,
								    TOutputImage> {
public:

  typedef TInputImage InputImageType;
  typedef TOutputImage OutputImageType;

  /** Standard class typedefs. */
  typedef CudaMeanImageFilter Self;
  typedef CudaImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(CudaMeanImageFilter,
	       CudaImageToImageFilter);

  typedef typename InputImageType::PixelType InputPixelType;
  typedef typename OutputImageType::PixelType OutputPixelType;

  typedef typename InputImageType::RegionType InputImageRegionType;
  typedef typename OutputImageType::RegionType OutputImageRegionType;

  typedef typename InputImageType::SizeType InputSizeType;

  /** Set the radius of the neighborhood used to compute the mean. */
  itkSetMacro(Radius, InputSizeType);

  /** Get the radius of the neighborhood used to compute the mean */
  itkGetConstReferenceMacro(Radius, InputSizeType);

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(InputHasNumericTraitsCheck,
                  (Concept::HasNumericTraits<InputPixelType>));
  /** End concept checking */
#endif

protected:
  CudaMeanImageFilter();
  ~CudaMeanImageFilter() {}
  void PrintSelf(std::ostream& os, Indent indent) const;
  void GenerateData();

private:
  CudaMeanImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  InputSizeType m_Radius;
};

} // end namespace itk
#ifndef ITK_MANUAL_INSTANTIATION
#include "CudaMeanImageFilter.txx"
#endif

#endif
