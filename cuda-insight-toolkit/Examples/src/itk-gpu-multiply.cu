/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itk-gpu-multiply.cu
  Language:  CUDA

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
/** \class itk-gpu-multiply.cu
 * \brief Cuda kernel code
 * \author Phillip Ward, Luke Parkinson, Daniel Micevski, Christopher
 * Share, Victorian Partnership for Advanced Computing (VPAC). 
 * Richard Beare, Monash University
 */

#include <stdio.h>
#include <stdlib.h>

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "CudaMultiplyImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "timer.h"

using namespace std;

int main(int argc, char **argv) {
	double start, end;

	// Pixel Types
	typedef float InputPixelType;
	typedef float OutputPixelType;
	const unsigned int Dimension = 2;
	int nFilters = atoi(argv[3]);

	// IO Types
	// typedef itk::RGBPixel< InputPixelType >       PixelType;
	typedef itk::Image<InputPixelType, Dimension> InputImageType;
	typedef itk::Image<OutputPixelType, Dimension> OutputImageType;
	typedef itk::ImageFileReader<InputImageType> ReaderType;
	typedef itk::ImageFileWriter<OutputImageType> WriterType;

	typedef itk::CudaMultiplyImageFilter<InputImageType, OutputImageType> FilterType;

	// Set Up Input File and Read Image
	ReaderType::Pointer reader1 = ReaderType::New();
	reader1->SetFileName(argv[1]);
	ReaderType::Pointer reader2 = ReaderType::New();
		reader2->SetFileName(argv[1]);

	try {
		reader1->Update();
		reader2->Update();
	} catch (itk::ExceptionObject exp) {
		cerr << "Reader caused problem." << endl;
		cerr << exp << endl;
		return 1;
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

	FilterType::Pointer filter[nFilters];
	filter[0] = FilterType::New();
	filter[0]->SetInput(0,reader1->GetOutput());
	filter[0]->SetInput(1,reader2->GetOutput());

	for (int i = 1; i < nFilters; ++i) {
		filter[i] = FilterType::New();
		filter[i]->SetInput(0,filter[i - 1]->GetOutput());
		filter[i]->SetInput(reader2->GetOutput());
	}

	try {
		start = getTime();
		filter[nFilters - 1]->Update();
		end = getTime();
		cout << end - start << endl;
	} catch (itk::ExceptionObject exp) {
		cerr << "Filter caused problem." << endl;
		cerr << exp << endl;
		return 1;
	}

	cout << argv[4] << endl;

	// Set Up Output File and Write Image
	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(argv[2]);
	writer->SetInput(filter[nFilters - 1]->GetOutput());

	try {
		writer->Update();
	} catch (itk::ExceptionObject exp) {
		cerr << "Filter caused problem." << endl;
		cerr << exp << endl;
		return 1;
	}

	return 0;
}

