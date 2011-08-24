/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    EclipseCompat.h
  Language:  C++

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
/*
** EclipseCompat.h
**
**  Created on: Jan 19, 2010
**      Author: lukep
*/

#ifndef ECLIPSECOMPAT_H_
#define ECLIPSECOMPAT_H_

#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __host__
#define __shared__
#endif

#endif /* ECLIPSECOMPAT_H_ */
