/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaNeighborhoodFilter.h
  Language:  C++

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#ifndef __itkCudaNeighborhoodFilter_h
#define __itkCudaNeighborhoodFilter_h

#include "itkImageToImageFilter.h"

namespace itk {

/** \class CudaNeighborhoodFilter
 * \brief Computes the ABS(x) pixel-wise
 *
 * \ingroup IntensityImageFilters  Multithreaded
 */

template<class TInputImage, class TOutputImage>
class ITK_EXPORT CudaNeighborhoodFilter: public ImageToImageFilter<TInputImage,
								   TOutputImage> {
public:

  typedef TInputImage InputImageType;
  typedef TOutputImage OutputImageType;

  /** Standard class typedefs. */
  typedef CudaNeighborhoodFilter Self;
  typedef ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)
    ;

  /** Runtime information support. */
  itkTypeMacro(CudaNeighborhoodFilter,
	       ImageToImageFilter)
    ;

  typedef typename InputImageType::PixelType InputPixelType;
  typedef typename OutputImageType::PixelType OutputPixelType;

  typedef typename InputImageType::RegionType InputImageRegionType;
  typedef typename OutputImageType::RegionType OutputImageRegionType;

  typedef typename InputImageType::SizeType InputSizeType;

  /** Set the radius of the neighborhood used to compute the mean. */
  itkSetMacro(Radius, InputSizeType)
    ;

  /** Get the radius of the neighborhood used to compute the mean */
  itkGetConstReferenceMacro(Radius, InputSizeType)
    ;

protected:
  CudaNeighborhoodFilter();
  ~CudaNeighborhoodFilter() {
  }
  void PrintSelf(std::ostream& os, Indent indent) const;
  void GenerateData();

private:
  CudaNeighborhoodFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  InputSizeType m_Radius;
};

} // end namespace itk
#ifndef ITK_MANUAL_INSTANTIATION
#include "CudaNeighborhoodFilter.txx"
#endif

#endif
