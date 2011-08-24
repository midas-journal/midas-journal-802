/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    CudaTextureFunctions.cu
  Language:  CUDA

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
/** \class CudaTextureFunctions.cu
 * \brief Cuda kernel code
 * \author Phillip Ward, Luke Parkinson, Daniel Micevski, Christopher
 * Share, Victorian Partnership for Advanced Computing (VPAC). 
 * Richard Beare, Monash University
 */

/*
 * File Name:    Cuda Texture Functions
 *
 * Author:        Phillip Ward
 * Creation Date: Monday, January 18 2010, 10:00 
 * Last Modified: Wednesday, February 23 2010, 16:35
 * 
 * File Description:
 *
 */
#include "EclipseCompat.h"
#include <stdio.h>
#include <cuda.h>
#include <cutil.h>

void copy3DHostToArray(float *_src, cudaArray *_dst, cudaExtent copy_extent, cudaPos src_offset, int3 imageDim)
{
 cudaMemcpy3DParms copyParams = {0};
 float *h_source = _src + src_offset.x + src_offset.y*imageDim.x + src_offset.z*imageDim.x*imageDim.y;
 copyParams.srcPtr = make_cudaPitchedPtr((void*)h_source, imageDim.x*sizeof(float), imageDim.x, imageDim.y);
 copyParams.dstArray = _dst;
 copyParams.kind = cudaMemcpyHostToDevice;
 copyParams.extent = copy_extent;

 CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
 CUT_CHECK_ERROR("Host -> Array Memcpy failed\n");
}

void copy3DDeviceToArray(float *_src, cudaArray *_dst, cudaExtent copy_extent, cudaPos src_offset, int3 imageDim)
{
 cudaMemcpy3DParms copyParams = {0};
 float *d_source = _src + src_offset.x + src_offset.y*imageDim.x + src_offset.z*imageDim.x*imageDim.y;
 copyParams.srcPtr = make_cudaPitchedPtr((void*)d_source, imageDim.x*sizeof(float), imageDim.x, imageDim.y);
 copyParams.dstArray = _dst;
 copyParams.kind = cudaMemcpyDeviceToDevice;
 copyParams.extent = copy_extent;

 CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
 CUT_CHECK_ERROR("Device -> Array Memcpy failed\n");
}

void copy3DMemToArray(cudaPitchedPtr _src, cudaArray *_dst, int3 imageDim)
{
 cudaMemcpy3DParms copyParams = {0};
 copyParams.srcPtr =  _src;
 copyParams.dstArray = _dst;
 copyParams.kind = cudaMemcpyDeviceToDevice;
 copyParams.extent = make_cudaExtent(imageDim.x, imageDim.y, imageDim.z);

 CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
 CUT_CHECK_ERROR("Mem -> Array Memcpy failed\n");
}
