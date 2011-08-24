/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaStatisticsImageFilter.h
  Language:  C++

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/


#ifndef __itkCudaStatisticsImageFilter_h
#define __itkCudaStatisticsImageFilter_h

#include "itkInPlaceImageFilter.h"
#include "itkNumericTraits.h"
#include "itkSimpleDataObjectDecorator.h"

namespace itk {

/** \class CudaStatisticsImageFilter
 * \brief Compute min. max, variance and mean of an Image.
 *
 * StatisticsImageFilter computes the minimum, maximum, sum, mean, variance
 * sigma of an image.  The filter needs all of its input image.  It
 * behaves as a filter with an input and output. Thus it can be inserted
 * in a pipline with other filters and the statistics will only be
 * recomputed if a downstream filter changes.
 *
 * The filter passes its input through unmodified.  The filter is
 * threaded. It computes statistics in each thread then combines them in
 * its AfterThreadedGenerate method.
 *
 * \author Phillip Ward, Luke Parkinson, Daniel Micevski, Christopher
 * Share, Victorian Partnership for Advanced Computing (VPAC). 
 * Richard Beare, Monash University
 *
 * \ingroup MathematicalStatisticsImageFilters CudaEnabled
 *
 * \sa ImageToImageFilter

 */

template<class TInputImage>
class ITK_EXPORT CudaStatisticsImageFilter: public ImageToImageFilter<
  TInputImage, TInputImage> {
public:

  typedef TInputImage InputImageType;

  /** Standard class typedefs. */
  typedef CudaStatisticsImageFilter Self;
  typedef ImageToImageFilter<TInputImage, TInputImage> Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(CudaStatisticsImageFilter,
	       ImageToImageFilter);

  typedef typename InputImageType::RegionType RegionType;
  typedef typename InputImageType::SizeType SizeType;
  typedef typename InputImageType::IndexType IndexType;
  typedef typename InputImageType::PixelType PixelType;

  /** Image related typedefs. */
  itkStaticConstMacro(ImageDimension, unsigned int,
		      TInputImage::ImageDimension );

  /** Type to use for computations. */
  typedef typename NumericTraits<PixelType>::RealType RealType;

  /** Smart Pointer type to a DataObject. */
  typedef typename DataObject::Pointer DataObjectPointer;

  /** Type of DataObjects used for scalar outputs */
  typedef SimpleDataObjectDecorator<RealType> RealObjectType;
  typedef SimpleDataObjectDecorator<PixelType> PixelObjectType;

  /** Return the computed Minimum. */
  PixelType GetMinimum() const {
    return this->GetMinimumOutput()->Get();
  }
  PixelObjectType* GetMinimumOutput();
  const PixelObjectType* GetMinimumOutput() const;

  /** Return the computed Maximum. */
  PixelType GetMaximum() const {
    return this->GetMaximumOutput()->Get();
  }
  PixelObjectType* GetMaximumOutput();
  const PixelObjectType* GetMaximumOutput() const;

  /** Return the computed Mean. */
  RealType GetMean() const {
    return this->GetMeanOutput()->Get();
  }
  RealObjectType* GetMeanOutput();
  const RealObjectType* GetMeanOutput() const;

  /** Return the computed Standard Deviation. */
  RealType GetSigma() const {
    return this->GetSigmaOutput()->Get();
  }
  RealObjectType* GetSigmaOutput();
  const RealObjectType* GetSigmaOutput() const;

  /** Return the computed Variance. */
  RealType GetVariance() const {
    return this->GetVarianceOutput()->Get();
  }
  RealObjectType* GetVarianceOutput();
  const RealObjectType* GetVarianceOutput() const;

  /** Return the compute Sum. */
  RealType GetSum() const {
    return this->GetSumOutput()->Get();
  }
  RealObjectType* GetSumOutput();
  const RealObjectType* GetSumOutput() const;

  /** Make a DataObject of the correct type to be used as the specified
	 * output. */
  virtual DataObjectPointer MakeOutput(unsigned int idx);

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(InputHasNumericTraitsCheck,
                  (Concept::HasNumericTraits<PixelType>));
  /** End concept checking */
#endif


protected:
  CudaStatisticsImageFilter();
  ~CudaStatisticsImageFilter() {
  }
  void PrintSelf(std::ostream& os, Indent indent) const;
  void GenerateData();
  void AllocateOutputs();

private:
  CudaStatisticsImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  RealType m_Sum;
  RealType m_SumOfSquares;
  float m_Count;
  PixelType m_Min;
  PixelType m_Max;
};

} // end namespace itk
#ifndef ITK_MANUAL_INSTANTIATION
#include "CudaStatisticsImageFilter.txx"
#endif

#endif
