/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    simple_perf_test.cxx
  Language:  C++

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
/** \class simple_perf_test.cxx
 * \brief Cuda kernel code
 * \author Phillip Ward, Luke Parkinson, Daniel Micevski, Christopher
 * Share, Victorian Partnership for Advanced Computing (VPAC). 
 * Richard Beare, Monash University
 */

#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "itkImage.h"

#include "CudaSubtractConstantFromImageFilter.h"
#include "itkSubtractConstantFromImageFilter.h"
#include "CudaAddConstantToImageFilter.h"
#include "itkAddConstantToImageFilter.h"
#include "CudaStatisticsImageFilter.h"
#include "timer.h"


int main(int argc, char **argv) 
{

  typedef float InputPixelType;
  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

  typedef itk::Image<InputPixelType, Dimension> InputImageType;
  typedef itk::Image<OutputPixelType, Dimension> OutputImageType;


  typedef itk::CudaAddConstantToImageFilter<InputImageType, InputImageType> AddType;
  typedef itk::SubtractConstantFromImageFilter<InputImageType, InputPixelType, InputImageType> SubType;
  {
  // dummy allocation to get the initialization out the way
  void *dInt;
  cudaMalloc(&dInt, sizeof(int)*1);
  cudaFree(dInt);
  }


  InputImageType::Pointer input = InputImageType::New();
  InputImageType::SizeType size;
  InputImageType::IndexType start;

  size.Fill(500);
  start.Fill(0);

  InputImageType::RegionType reg;
  reg.SetSize(size);
  reg.SetIndex(start);

  input->SetRegions(reg);
  input->Allocate();
  input->FillBuffer(0);

  const unsigned iterations = 200;

  double startcpu=0, endcpu=0;
  double startgpu=0, endgpu=0;

  AddType::Pointer adder = AddType::New();
  adder->SetConstant(2);
  InputImageType::Pointer res;
  startgpu = getTime();

  for (unsigned i=0;i<iterations;i++)
    {
    adder->SetInput(input);
    adder->Modified();
    res = adder->GetOutput();
    res->Update();
    res->DisconnectPipeline();
    input=res;
    }
  endgpu = getTime();
  SubType::Pointer sub = SubType::New();
  sub->SetConstant(2);

  startcpu = getTime();
  for (unsigned i=0;i<iterations;i++)
    {
    sub->SetInput(input);
    sub->Modified();
    res = sub->GetOutput();
    res->Update();
    res->DisconnectPipeline();
    input=res;
    }
  endcpu = getTime();
  std::cout << endgpu - startgpu << " " << endcpu - startcpu << std::endl;
  return EXIT_SUCCESS;
}
