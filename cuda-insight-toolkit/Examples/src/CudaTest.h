/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaTest.h
  Language:  C++

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __CudaTest_h
#define __CudaTest_h

using namespace std;
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "timer.h"

template <class FilterType, class InputImageType, class OutputImageType>
int CudaTest2(int nFilters, bool InPlace, char * in1, char * in2, char * out1)
{
  double start, end;
  typedef typename InputImageType::PixelType InputPixelType;
  typedef typename OutputImageType::PixelType OutputPixelType;
  const unsigned int Dimension = InputImageType::ImageDimension;

  // IO Types
  // typedef itk::RGBPixel< InputPixelType >       PixelType;
  typedef typename itk::ImageFileReader<InputImageType> ReaderType;
  typedef typename itk::ImageFileWriter<OutputImageType> WriterType;

  // Set Up Input File and Read Image
  typename ReaderType::Pointer reader1 = ReaderType::New();
  reader1->SetFileName(in1);
  typename ReaderType::Pointer reader2 = ReaderType::New();
  reader2->SetFileName(in2);

  try
    {
    reader1->Update();
    reader2->Update();
    }
  catch (itk::ExceptionObject exp)
    {
    cerr << "Reader caused problem." << endl;
    cerr << exp << endl;
    return EXIT_FAILURE;
    }

  for (unsigned int i = 0; i < 3; ++i) {
  if (i < Dimension) {
  cout
    << reader1->GetOutput()->GetLargestPossibleRegion().GetSize()[i]
    << ", ";
  } else {
  cout << 1 << ", ";
  }
  }

  typename FilterType::Pointer filter[nFilters];
  filter[0] = FilterType::New();
  filter[0]->SetInput(0,reader1->GetOutput());
  filter[0]->SetInput(1,reader2->GetOutput());
  filter[0]->SetInPlace(InPlace);
  for (int i = 1; i < nFilters; ++i)
    {
    filter[i] = FilterType::New();
    filter[i]->SetInput(0,filter[i - 1]->GetOutput());
    filter[i]->SetInput(reader2->GetOutput());
    }

  try
    {
    start = getTime();
    filter[nFilters - 1]->Update();
    end = getTime();
    cout << end - start << endl;
    }
  catch (itk::ExceptionObject exp)
    {
    cerr << "Filter caused problem." << endl;
    cerr << exp << endl;
    return EXIT_FAILURE;
    }

  // Set Up Output File and Write Image
  typename WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(out1);
  writer->SetInput(filter[nFilters - 1]->GetOutput());

  try
    {
    writer->Update();
    }
  catch (itk::ExceptionObject exp)
    {
    cerr << "Filter caused problem." << endl;
    cerr << exp << endl;
    return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;


}

template <class FilterType, class InputImageType, class OutputImageType>
int CudaTest1(int nFilters, bool InPlace, char * in1, char * out1)
{
  double start, end;
  typedef typename InputImageType::PixelType InputPixelType;
  typedef typename OutputImageType::PixelType OutputPixelType;
  const unsigned int Dimension = InputImageType::ImageDimension;

  // IO Types
  // typedef itk::RGBPixel< InputPixelType >       PixelType;
  typedef typename itk::ImageFileReader<InputImageType> ReaderType;
  typedef typename itk::ImageFileWriter<OutputImageType> WriterType;

  // Set Up Input File and Read Image
  typename ReaderType::Pointer reader1 = ReaderType::New();
  reader1->SetFileName(in1);

  try
    {
    reader1->Update();
    }
  catch (itk::ExceptionObject exp)
    {
    cerr << "Reader caused problem." << endl;
    cerr << exp << endl;
    return EXIT_FAILURE;
    }

  for (unsigned int i = 0; i < 3; ++i) {
  if (i < Dimension) {
  cout
    << reader1->GetOutput()->GetLargestPossibleRegion().GetSize()[i]
    << ", ";
  } else {
  cout << 1 << ", ";
  }
  }

  typename FilterType::Pointer filter[nFilters];
  filter[0] = FilterType::New();
  filter[0]->SetInput(0,reader1->GetOutput());
  filter[0]->SetInPlace(InPlace);
  for (int i = 1; i < nFilters; ++i)
    {
    filter[i] = FilterType::New();
    filter[i]->SetInput(0,filter[i - 1]->GetOutput());
    }

  try
    {
    start = getTime();
    filter[nFilters - 1]->Update();
    end = getTime();
    cout << end - start << endl;
    }
  catch (itk::ExceptionObject exp)
    {
    cerr << "Filter caused problem." << endl;
    cerr << exp << endl;
    return EXIT_FAILURE;
    }

  // Set Up Output File and Write Image
  typename WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(out1);
  writer->SetInput(filter[nFilters - 1]->GetOutput());

  try
    {
    writer->Update();
    }
  catch (itk::ExceptionObject exp)
    {
    cerr << "Filter caused problem." << endl;
    cerr << exp << endl;
    return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;


}

template <class FilterType, class InputImageType, class OutputImageType>
int CudaTest1a(bool InPlace, char * in1, char * out1,
	       typename FilterType::Pointer filter)
{
  double start, end;
  typedef typename InputImageType::PixelType InputPixelType;
  typedef typename OutputImageType::PixelType OutputPixelType;
  const unsigned int Dimension = InputImageType::ImageDimension;

  // IO Types
  // typedef itk::RGBPixel< InputPixelType >       PixelType;
  typedef typename itk::ImageFileReader<InputImageType> ReaderType;
  typedef typename itk::ImageFileWriter<OutputImageType> WriterType;

  // Set Up Input File and Read Image
  typename ReaderType::Pointer reader1 = ReaderType::New();
  reader1->SetFileName(in1);

  try
    {
    reader1->Update();
    }
  catch (itk::ExceptionObject exp)
    {
    cerr << "Reader caused problem." << endl;
    cerr << exp << endl;
    return EXIT_FAILURE;
    }

  for (unsigned int i = 0; i < 3; ++i) {
  if (i < Dimension) {
  cout
    << reader1->GetOutput()->GetLargestPossibleRegion().GetSize()[i]
    << ", ";
  } else {
  cout << 1 << ", ";
  }
  }


  filter->SetInput(0,reader1->GetOutput());
  filter->SetInPlace(InPlace);

  try
    {
    start = getTime();
    filter->Update();
    end = getTime();
    cout << end - start << endl;
    }
  catch (itk::ExceptionObject exp)
    {
    cerr << "Filter caused problem." << endl;
    cerr << exp << endl;
    return EXIT_FAILURE;
    }

  // Set Up Output File and Write Image
  typename WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(out1);
  writer->SetInput(filter->GetOutput());

  try
    {
    writer->Update();
    }
  catch (itk::ExceptionObject exp)
    {
    cerr << "Filter caused problem." << endl;
    cerr << exp << endl;
    return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;


}

template <class FilterType, class InputImageType, class OutputImageType>
int CudaTest1b(char * in1, char * out1,
	       typename FilterType::Pointer filter)
{
  double start, end;
  typedef typename InputImageType::PixelType InputPixelType;
  typedef typename OutputImageType::PixelType OutputPixelType;
  const unsigned int Dimension = InputImageType::ImageDimension;

  // IO Types
  // typedef itk::RGBPixel< InputPixelType >       PixelType;
  typedef typename itk::ImageFileReader<InputImageType> ReaderType;
  typedef typename itk::ImageFileWriter<OutputImageType> WriterType;

  // Set Up Input File and Read Image
  typename ReaderType::Pointer reader1 = ReaderType::New();
  reader1->SetFileName(in1);

  try
    {
    reader1->Update();
    }
  catch (itk::ExceptionObject exp)
    {
    cerr << "Reader caused problem." << endl;
    cerr << exp << endl;
    return EXIT_FAILURE;
    }

  for (unsigned int i = 0; i < 3; ++i) {
  if (i < Dimension) {
  cout
    << reader1->GetOutput()->GetLargestPossibleRegion().GetSize()[i]
    << ", ";
  } else {
  cout << 1 << ", ";
  }
  }


  filter->SetInput(0,reader1->GetOutput());
  try
    {
    start = getTime();
    filter->Update();
    end = getTime();
    cout << end - start << endl;
    }
  catch (itk::ExceptionObject exp)
    {
    cerr << "Filter caused problem." << endl;
    cerr << exp << endl;
    return EXIT_FAILURE;
    }

  // Set Up Output File and Write Image
  typename WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(out1);
  writer->SetInput(filter->GetOutput());

  try
    {
    writer->Update();
    }
  catch (itk::ExceptionObject exp)
    {
    cerr << "Filter caused problem." << endl;
    cerr << exp << endl;
    return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;


}

#endif
