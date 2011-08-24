/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaFilter.h
  Language:  C++

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/


#ifndef __itkCudaFilter_h
#define __itkCudaFilter_h

#include "itkImageToImageFilter.h"

namespace itk
{

/** \class CudaFilter
 * \brief Computes the ABS(x) pixel-wise
 *
 * \author Phillip Ward, Victorian Partnership for Advanced Computing (VPAC)
 *
 * \ingroup IntensityImageFilters  Multithreaded
 *
 * \sa ImageToImageFilter
 */

template <class TInputImage, class TOutputImage>
class ITK_EXPORT CudaFilter :
    public
ImageToImageFilter<TInputImage, TOutputImage >
{
public:

  typedef TInputImage                 InputImageType;
  typedef TOutputImage                OutputImageType;

  /** Standard class typedefs. */
  typedef CudaFilter  Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage >
    Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(CudaFilter,
               ImageToImageFilter);

  typedef typename InputImageType::PixelType   InputPixelType;
  typedef typename OutputImageType::PixelType  OutputPixelType;

  typedef typename InputImageType::RegionType  InputImageRegionType;
  typedef typename OutputImageType::RegionType OutputImageRegionType;

  typedef typename InputImageType::SizeType    InputSizeType;

protected:
  CudaFilter();
  ~CudaFilter() {}
  void PrintSelf(std::ostream& os, Indent indent) const;
  void GenerateData();

private:
  CudaFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk
#ifndef ITK_MANUAL_INSTANTIATION
#include "CudaFilter.txx"
#endif

#endif
