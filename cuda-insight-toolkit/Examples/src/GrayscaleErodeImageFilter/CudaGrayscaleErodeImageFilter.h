/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaGrayscaleErodeImageFilter.h
  Language:  C++

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#ifndef __itkCudaGrayscaleErodeImageFilter_h
#define __itkCudaGrayscaleErodeImageFilter_h

#include "CudaImageToImageFilter.h"
#include "itkNeighborhood.h"

namespace itk {

/** \class CudaGrayscaleErodeImageFilter
 * \brief gray scale erosion of an image
 *
 * Erode an image using grayscale morphology. Erosion takes the
 * minimum of all the pixels identified by the structuring element.
 *
 * The structuring element is assumed to be composed of binary
 * values (zero or one). Only elements of the structuring element
 * having values > 0 are candidates for affecting the center pixel.
 *
 * For the each input image pixel,
 *   - NeighborhoodIterator gives neighbors of the pixel.
 *   - Evaluate() member function returns the minimum value among
 *     the image neighbors where the kernel has elements > 0
 *   - Replace the original value with the min value
 *
  * \author Phillip Ward, Luke Parkinson, Daniel Micevski, Christopher
 * Share, Victorian Partnership for Advanced Computing (VPAC). 
 * Richard Beare, Monash University
 *
 *
 * \sa CudaImageToImageFilter
 * \ingroup ImageEnhancement  MathematicalMorphologyImageFilters  CudaEnabled
 */


template<class TInputImage, class TOutputImage, class TKernel>
class ITK_EXPORT CudaGrayscaleErodeImageFilter: public CudaImageToImageFilter<TInputImage,
									  TOutputImage> {
public:

  /** Standard class typedefs. */
  typedef CudaGrayscaleErodeImageFilter Self;
  typedef CudaImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)	;

  /** Runtime information support. */
  itkTypeMacro(CudaGrayscaleErodeImageFilter,
	       CudaImageToImageFilter)	;

  typedef TInputImage                                   InputImageType;
  typedef TOutputImage                                  OutputImageType;
  typedef typename TInputImage::RegionType              RegionType;
  typedef typename TInputImage::SizeType                SizeType;
  typedef typename TInputImage::IndexType               IndexType;
  typedef typename TInputImage::PixelType               InputPixelType;
  typedef typename Superclass::OutputImageRegionType    OutputImageRegionType;
  typedef typename TOutputImage::PixelType              OutputPixelType;
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
  itkConceptMacro(InputConvertibleToOutputCheck,
		  (Concept::Convertible<InputPixelType, OutputPixelType>));
  itkConceptMacro(SameDimensionCheck1,
		  (Concept::SameDimension<InputImageDimension, OutputImageDimension>));
  itkConceptMacro(SameDimensionCheck2,
		  (Concept::SameDimension<InputImageDimension, KernelDimension>));
  itkConceptMacro(InputLessThanComparableCheck,
		  (Concept::LessThanComparable<InputPixelType>));
  itkConceptMacro(KernelGreaterThanComparableCheck,
		  (Concept::GreaterThanComparable<KernelPixelType>));
  /** End concept checking */
#endif


protected:
  CudaGrayscaleErodeImageFilter();
  ~CudaGrayscaleErodeImageFilter() {
  }
  void PrintSelf(std::ostream& os, Indent indent) const;
  void GenerateData();

private:
  CudaGrayscaleErodeImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  /** kernel or structuring element to use. */
  KernelType m_Kernel;
};

} // end namespace itk
#ifndef ITK_MANUAL_INSTANTIATION
#include "CudaGrayscaleErodeImageFilter.txx"
#endif

#endif
