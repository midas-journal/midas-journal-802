/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaGrayscaleMorphologicalOpeningImageFilter.h
  Language:  C++

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#ifndef __itkCudaGrayscaleMorphologicalOpeningImageFilter_h
#define __itkCudaGrayscaleMorphologicalOpeningImageFilter_h

#include "CudaImageToImageFilter.h"
#include "itkProgressAccumulator.h"
#include "CudaGrayscaleDilateImageFilter.h"
#include "CudaGrayscaleErodeImageFilter.h"

namespace itk {

/**
 * \class CudaGrayscaleMorphologicalOpeningImageFilter
 * \brief gray scale morphological opening of an image.
 *
 * This filter preserves regions, in the foreground, that can
 * completely contain the structuring element. At the same time,
 * this filter eliminates all other regions of foreground
 * pixels. The morphological opening of an image "f"
 * is defined as:
 * Opening(f) = Dilation(Erosion(f)).
 *
 * The structuring element is assumed to be composed of binary
 * values (zero or one). Only elements of the structuring element
 * having values > 0 are candidates for affecting the center pixel.
 *
 *
 * \author Phillip Ward, Luke Parkinson, Daniel Micevski, Christopher
 * Share, Victorian Partnership for Advanced Computing (VPAC). 
 * Richard Beare, Monash University
 *
 * \sa CudaImageToImageFilter
 * \ingroup ImageEnhancement  MathematicalMorphologyImageFilters  CudaEnabled
 */

template<class TInputImage, class TOutputImage, class TKernel>
class ITK_EXPORT CudaGrayscaleMorphologicalOpeningImageFilter: public CudaImageToImageFilter<TInputImage,
											 TOutputImage> {
public:

  /** Standard class typedefs. */
  typedef CudaGrayscaleMorphologicalOpeningImageFilter Self;
  typedef CudaImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)	;

  /** Runtime information support. */
  itkTypeMacro(CudaGrayscaleMorphologicalOpeningImageFilter,
	       CudaImageToImageFilter)	;

  typedef TInputImage                                   InputImageType;
  typedef TOutputImage                                  OutputImageType;
  typedef typename TInputImage::RegionType              RegionType;
  typedef typename TInputImage::SizeType                SizeType;
  typedef typename TInputImage::IndexType               IndexType;
  typedef typename TInputImage::PixelType               PixelType;
  typedef typename Superclass::OutputImageRegionType    OutputImageRegionType;

  /** Image related typedefs. */
  itkStaticConstMacro(ImageDimension, unsigned int, TInputImage::ImageDimension);

  /** Kernel typedef. */
  typedef TKernel KernelType;
  typedef typename TKernel::PixelType            KernelPixelType;

  /** n-dimensional Kernel radius. */
  typedef typename KernelType::SizeType RadiusType;

  /** Set kernel (structuring element). */
  itkSetMacro(Kernel, KernelType);

  /** Get the kernel (structuring element). */
  itkGetConstReferenceMacro(Kernel, KernelType);

  /** ImageDimension constants */
  itkStaticConstMacro(InputImageDimension, unsigned int,
		      TInputImage::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int,
		      TOutputImage::ImageDimension);
  itkStaticConstMacro(KernelDimension, unsigned int,
		      TKernel::NeighborhoodDimension);


#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(SameTypeCheck,
		  (Concept::SameType<PixelType, typename TOutputImage::PixelType>));
  itkConceptMacro(SameDimensionCheck1,
		  (Concept::SameDimension<InputImageDimension, OutputImageDimension>));
  itkConceptMacro(SameDimensionCheck2,
		  (Concept::SameDimension<InputImageDimension, KernelDimension>));
  itkConceptMacro(InputLessThanComparableCheck,
		  (Concept::LessThanComparable<PixelType>));
  itkConceptMacro(InputGreaterThanComparableCheck,
		  (Concept::GreaterThanComparable<PixelType>));
  itkConceptMacro(KernelGreaterThanComparableCheck,
		  (Concept::GreaterThanComparable<KernelPixelType>));
  /** End concept checking */
#endif

protected:
  CudaGrayscaleMorphologicalOpeningImageFilter();
  ~CudaGrayscaleMorphologicalOpeningImageFilter() {
  }
  void PrintSelf(std::ostream& os, Indent indent) const;
  void GenerateData();

private:
  CudaGrayscaleMorphologicalOpeningImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  /** kernel or structuring element to use. */
  KernelType m_Kernel;
};

} // end namespace itk
#ifndef ITK_MANUAL_INSTANTIATION
#include "CudaGrayscaleMorphologicalOpeningImageFilter.txx"
#endif

#endif
