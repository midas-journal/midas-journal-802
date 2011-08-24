/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaBinaryThresholdImageFilter.h
  Language:  C++

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#ifndef __CudaBinaryThresholdImageFilter_h
#define __CudaBinaryThresholdImageFilter_h

#include "CudaInPlaceImageFilter.h"

namespace itk
{

/** \class CudaBinaryThresholdImageFilter
 *
 * \brief Binarize an input image by thresholding.
 *
 * This filter produces an output image whose pixels
 * are either one of two values ( OutsideValue or InsideValue ),
 * depending on whether the corresponding input image pixels
 * lie between the two thresholds ( LowerThreshold and UpperThreshold ).
 * Values equal to either threshold is considered to be between the thresholds.
 *
 * More precisely
 * \f[ Output(x_i) =
       \begin{cases}
         InsideValue  & \text{if $LowerThreshold \leq x_i \leq UpperThreshold$} \\
         OutsideValue & \text{otherwise}
        \end{cases}
   \f]
 *
 * This filter is templated over the input image type
 * and the output image type.
 *
 * The filter expect both images to have the same number of dimensions.
 *
 * The default values for LowerThreshold and UpperThreshold are:
 * LowerThreshold = NumericTraits<TInput>::NonpositiveMin();
 * UpperThreshold = NumericTraits<TInput>::max();
 * Therefore, generally only one of these needs to be set, depending
 * on whether the user wants to threshold above or below the desired threshold.
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
class ITK_EXPORT CudaBinaryThresholdImageFilter :
    public
CudaInPlaceImageFilter<TInputImage, TOutputImage >
{
public:

  typedef TInputImage                 InputImageType;
  typedef TOutputImage                OutputImageType;


  typedef CudaBinaryThresholdImageFilter           Self;
  typedef CudaInPlaceImageFilter<TInputImage,TOutputImage >
    Superclass;
  typedef SmartPointer<Self>          Pointer;
  typedef SmartPointer<const Self>    ConstPointer;

  itkNewMacro(Self);

  itkTypeMacro(CudaBinaryThresholdImageFilter, CudaInPlaceImageFilter);

  typedef typename InputImageType::PixelType   InputPixelType;
  typedef typename OutputImageType::PixelType  OutputPixelType;

  typedef typename InputImageType::RegionType  InputImageRegionType;
  typedef typename OutputImageType::RegionType OutputImageRegionType;

  typedef typename InputImageType::SizeType    InputSizeType;

  itkSetMacro(LowerThreshold, InputPixelType);
  itkSetMacro(UpperThreshold, InputPixelType);
  itkSetMacro(InsideValue, OutputPixelType);
  itkSetMacro(OutsideValue, OutputPixelType);

  itkGetConstReferenceMacro(LowerThreshold, InputPixelType);
  itkGetConstReferenceMacro(UpperThreshold, InputPixelType);
  itkGetConstReferenceMacro(InsideValue, OutputPixelType);
  itkGetConstReferenceMacro(OutsideValue, OutputPixelType);

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(OutputEqualityComparableCheck,
                  (Concept::EqualityComparable<OutputPixelType>));
  itkConceptMacro(InputPixelTypeComparable,
                  (Concept::Comparable<InputPixelType>));
  itkConceptMacro(InputOStreamWritableCheck,
                  (Concept::OStreamWritable<InputPixelType>));
  itkConceptMacro(OutputOStreamWritableCheck,
                  (Concept::OStreamWritable<OutputPixelType>));
  /** End concept checking */
#endif


protected:
  CudaBinaryThresholdImageFilter();
  ~CudaBinaryThresholdImageFilter() {}
  void PrintSelf(std::ostream& os, Indent indent) const;
  void GenerateData();

private:
  CudaBinaryThresholdImageFilter(const Self&);
  void operator=(const Self&);

  InputPixelType m_LowerThreshold;
  InputPixelType m_UpperThreshold;
  OutputPixelType m_InsideValue;
  OutputPixelType m_OutsideValue;
};
}

#ifndef ITK_MANUAL_INSTANTIATION
#include "CudaBinaryThresholdImageFilter.txx"
#endif

#endif
