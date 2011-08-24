
##############################################################################
# There are two ways to add include directories to the NVCC command
# line:

IF (CITK_USE_CUDA)
set(CUDA_PROPAGATE_HOST_FLAGS ON)


# The cuda_include_directories adds paths to only cuda compilation.
CUDA_INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/thrust

INCLUDE_DIRECTORIES(  
	${CUDA_INCLUDE_DIRECTORIES}
	)
# Set CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE when you want to add the same .cu
# file to multiple targets.
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
)

ENDIF(CITK_USE_CUDA)

# The include_directories adds paths to both C/C++ compilation in the native
# compiler and cuda compilation in NVCC.  Note that CUDA_INCLUDE_DIRS is added
# automatically by CUDA_ADD_EXECUTABLE and CUDA_ADD_LIBRARY.

INCLUDE_DIRECTORIES(
   AbsImageFilter
   AddImageFilter
   AddConstantToImageFilter
   BinaryThresholdImageFilter
   DivideImageFilter
   DivideByConstantImageFilter
   GrayscaleDilateImageFilter
   GrayscaleErodeImageFilter
   GrayscaleMorphologicalClosingImageFilter
   GrayscaleMorphologicalOpeningImageFilter
   MaximumImageFilter
   MeanImageFilter
   MinimumImageFilter
   MultiplyImageFilter
   MultiplyByConstantImageFilter
   NeighborhoodFilter
   RescaleIntensityImageFilter
   StatisticsImageFilter
   SubtractImageFilter
   SubtractConstantFromImageFilter
   CudaFunctions
)

##############################################################################
# There are four ways to compile source files with NVCC. 

IF (CITK_USE_CUDA)

# Use one executable only.
CUDA_ADD_EXECUTABLE(ITK-GPU-OPEN 
 GrayscaleDilateImageFilter/CudaGrayscaleDilateImageFilterKernel.cu
 GrayscaleErodeImageFilter/CudaGrayscaleErodeImageFilterKernel.cu
 itk-gpu-open.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-OPEN
  ITKBasicFilters ITKCommon ITKIO
)
 CUDA_ADD_EXECUTABLE(ITK-GPU-CLOSE 
 GrayscaleDilateImageFilter/CudaGrayscaleDilateImageFilterKernel.cu
 GrayscaleErodeImageFilter/CudaGrayscaleErodeImageFilterKernel.cu
 itk-gpu-close.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-CLOSE
  ITKBasicFilters ITKCommon ITKIO
)
  CUDA_ADD_EXECUTABLE(ITK-GPU-ERODE 
 GrayscaleErodeImageFilter/CudaGrayscaleErodeImageFilterKernel.cu
 itk-gpu-erode.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-ERODE
  ITKBasicFilters ITKCommon ITKIO
)
  CUDA_ADD_EXECUTABLE(ITK-GPU-DILATE 
 GrayscaleDilateImageFilter/CudaGrayscaleDilateImageFilterKernel.cu
 itk-gpu-dilate.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-DILATE
  ITKBasicFilters ITKCommon ITKIO
)
  CUDA_ADD_EXECUTABLE(ITK-GPU-ABS 
 AbsImageFilter/CudaAbsImageFilterKernel.cu
 itk-gpu-abs.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-ABS
  ITKBasicFilters ITKCommon ITKIO 
)
   CUDA_ADD_EXECUTABLE(ITK-GPU-ADD 
 AddImageFilter/CudaAddImageFilterKernel.cu
 itk-gpu-add.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-ADD
  ITKBasicFilters ITKCommon ITKIO
)
   CUDA_ADD_EXECUTABLE(ITK-GPU-ADDC	
 AddConstantToImageFilter/CudaAddConstantToImageFilterKernel.cu
 itk-gpu-addc.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-ADDC
  ITKBasicFilters ITKCommon ITKIO
)
 CUDA_ADD_EXECUTABLE(ITK-GPU-BINARYT 
 BinaryThresholdImageFilter/CudaBinaryThresholdImageFilterKernel.cu
 itk-gpu-binarythreshold.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-BINARYT
  ITKBasicFilters ITKCommon ITKIO
)
   CUDA_ADD_EXECUTABLE(ITK-GPU-DIVIDE 
 DivideImageFilter/CudaDivideImageFilterKernel.cu
 itk-gpu-divide.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-DIVIDE
  ITKBasicFilters ITKCommon ITKIO
)

CUDA_ADD_EXECUTABLE(ITK-GPU-DIVIDEC 
 DivideByConstantImageFilter/CudaDivideByConstantImageFilterKernel.cu
 itk-gpu-dividec.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-DIVIDEC
  ITKBasicFilters ITKCommon ITKIO
)

 CUDA_ADD_EXECUTABLE(ITK-GPU-MAXIMUM 
 MaximumImageFilter/CudaMaximumImageFilterKernel.cu
 itk-gpu-maximum.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-MAXIMUM
  ITKBasicFilters ITKCommon ITKIO
)
  CUDA_ADD_EXECUTABLE(ITK-GPU-MINIMUM 
 MinimumImageFilter/CudaMinimumImageFilterKernel.cu
 itk-gpu-minimum.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-MINIMUM
  ITKBasicFilters ITKCommon ITKIO
)
  CUDA_ADD_EXECUTABLE(ITK-GPU-MEAN 
 MeanImageFilter/CudaMeanImageFilterKernel.cu
 itk-gpu-mean.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-MEAN
  ITKBasicFilters ITKCommon ITKIO
)
CUDA_ADD_EXECUTABLE(ITK-GPU-MULTIPLY 
 MultiplyImageFilter/CudaMultiplyImageFilterKernel.cu
 itk-gpu-multiply.cxx
 )
TARGET_LINK_LIBRARIES ( ITK-GPU-MULTIPLY
 ITKBasicFilters ITKCommon ITKIO
)

CUDA_ADD_EXECUTABLE(ITK-GPU-MULTIPLYC 
 MultiplyByConstantImageFilter/CudaMultiplyByConstantImageFilterKernel.cu
 itk-gpu-multiplyc.cxx
 )
TARGET_LINK_LIBRARIES ( ITK-GPU-MULTIPLYC
 ITKBasicFilters ITKCommon ITKIO
)

 CUDA_ADD_EXECUTABLE(ITK-GPU-RESCALE 
 RescaleIntensityImageFilter/CudaRescaleIntensityImageFilterKernel.cu
 itk-gpu-rescaleintensity.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-RESCALE
  ITKBasicFilters ITKCommon ITKIO
)
 CUDA_ADD_EXECUTABLE(ITK-GPU-STATISTICS 
 StatisticsImageFilter/CudaStatisticsImageFilterKernel.cu
 itk-gpu-statistics.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-STATISTICS
  ITKBasicFilters ITKCommon ITKIO
)
   CUDA_ADD_EXECUTABLE(ITK-GPU-SUBTRACT 
 SubtractImageFilter/CudaSubtractImageFilterKernel.cu
 itk-gpu-subtract.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-SUBTRACT
  ITKBasicFilters ITKCommon ITKIO
)

CUDA_ADD_EXECUTABLE(ITK-GPU-SUBTRACTC 
 SubtractConstantFromImageFilter/CudaSubtractConstantFromImageFilterKernel.cu
 itk-gpu-subtractc.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-GPU-SUBTRACTC
  ITKBasicFilters ITKCommon ITKIO
)

CUDA_ADD_EXECUTABLE(ITK-SIMPLE-PERF
 SubtractConstantFromImageFilter/CudaSubtractConstantFromImageFilterKernel.cu
 AddConstantToImageFilter/CudaAddConstantToImageFilterKernel.cu
 StatisticsImageFilter/CudaStatisticsImageFilterKernel.cu
 simple_perf_test.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-SIMPLE-PERF
  ITKBasicFilters ITKCommon ITKIO
)


CUDA_ADD_EXECUTABLE(ITK-MEAN-PERF
 MeanImageFilter/CudaMeanImageFilterKernel.cu
 mean_perf_test.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-MEAN-PERF
  ITKBasicFilters ITKCommon ITKIO
)

ENDIF(CITK_USE_CUDA)

ADD_EXECUTABLE(ITK-CPU-OPEN 
 itk-cpu-open.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-OPEN
  ITKBasicFilters ITKCommon ITKIO rt
)
ADD_EXECUTABLE(ITK-CPU-CLOSE 
 itk-cpu-close.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-CLOSE
  ITKBasicFilters ITKCommon ITKIO rt
)
 ADD_EXECUTABLE(ITK-CPU-ERODE 
itk-cpu-erode.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-ERODE
  ITKBasicFilters ITKCommon ITKIO rt
)
 ADD_EXECUTABLE(ITK-CPU-DILATE 
 itk-cpu-dilate.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-DILATE
  ITKBasicFilters ITKCommon ITKIO rt
)
 ADD_EXECUTABLE(ITK-CPU-ABS 
 itk-cpu-abs.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-ABS
  ITKBasicFilters ITKCommon ITKIO rt
)
 ADD_EXECUTABLE(ITK-CPU-ADD 
 itk-cpu-add.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-ADD
  ITKBasicFilters ITKCommon ITKIO rt
)
 ADD_EXECUTABLE(ITK-CPU-ADDC
 itk-cpu-addc.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-ADDC
  ITKBasicFilters ITKCommon ITKIO rt
)
 ADD_EXECUTABLE(ITK-CPU-BINARYT 
 itk-cpu-binarythreshold.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-BINARYT
  ITKBasicFilters ITKCommon ITKIO rt
)
 ADD_EXECUTABLE(ITK-CPU-DIVIDE 
 itk-cpu-divide.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-DIVIDE
  ITKBasicFilters ITKCommon ITKIO rt
)
 ADD_EXECUTABLE(ITK-CPU-DIVIDEC 
 itk-cpu-dividec.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-DIVIDEC
  ITKBasicFilters ITKCommon ITKIO rt
)

 ADD_EXECUTABLE(ITK-CPU-MAXIMUM 
 itk-cpu-maximum.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-MAXIMUM
  ITKBasicFilters ITKCommon ITKIO rt
)
 ADD_EXECUTABLE(ITK-CPU-MINIMUM 
 itk-cpu-minimum.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-MINIMUM
  ITKBasicFilters ITKCommon ITKIO rt
)
 ADD_EXECUTABLE(ITK-CPU-MEAN 
 itk-cpu-mean.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-MEAN
  ITKBasicFilters ITKCommon ITKIO rt
)
 ADD_EXECUTABLE(ITK-CPU-MULTIPLY 
 itk-cpu-multiply.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-MULTIPLY
  ITKBasicFilters ITKCommon ITKIO rt
)

 ADD_EXECUTABLE(ITK-CPU-MULTIPLYC 
 itk-cpu-multiplyc.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-MULTIPLYC
  ITKBasicFilters ITKCommon ITKIO rt
)

 ADD_EXECUTABLE(ITK-CPU-RESCALE 
 itk-cpu-rescaleintensity.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-RESCALE
  ITKBasicFilters ITKCommon ITKIO rt
)
 ADD_EXECUTABLE(ITK-CPU-STATISTICS 
 itk-cpu-statistics.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-STATISTICS
  ITKBasicFilters ITKCommon ITKIO rt
)
 ADD_EXECUTABLE(ITK-CPU-SUBTRACT 
 itk-cpu-subtract.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-SUBTRACT
  ITKBasicFilters ITKCommon ITKIO rt
)

 ADD_EXECUTABLE(ITK-CPU-SUBTRACTC 
 itk-cpu-subtractc.cxx
 )
 TARGET_LINK_LIBRARIES ( ITK-CPU-SUBTRACTC
  ITKBasicFilters ITKCommon ITKIO rt
)

TARGET_LINK_LIBRARIES ( ITK-CPU-SUBTRACTC
  ITKBasicFilters ITKCommon ITKIO rt
)


ADD_EXECUTABLE(ImageCompare ImageCompare.cxx)
TARGET_LINK_LIBRARIES(ImageCompare ITKCommon ITKIO)

# Or compile the cuda code into a shared library.

# Anything other than -D or /D is not passed along to nvcc.
#add_definitions(-DMULTIPLIER=2)
#
## You can set BUILD_SHARED_LIBS or you can pass STATIC, SHARED, or MODULE to
## CUDA_ADD_LIBRARY.  Remember that BUILD_SHARED_LIBS is only respected by
## CUDA_ADD_LIBRARY.  If you use CUDA_COMPILE or CUDA_WRAP_SRCS you must specify
## SHARED or STATIC as a parameter.
#set(BUILD_SHARED_LIBS ON)
#set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
#CUDA_ADD_LIBRARY(test_lib
#  test_lib.cu
#  external_dependency.h
#  # SHARED
#  # STATIC
#  OPTIONS "-DSTUFF=blah blah"
#  RELEASE --use_fast_math -DNDEBUG
#  DEBUG -g -DDEBUG
#  )
#
## Then link the shared library to another executable.
#ADD_EXECUTABLE(lib_example
#  main_for_lib.cc
#  )
#
## Specify the dependency.
#TARGET_LINK_LIBRARIES(lib_example
#  test_lib
#  )
#
## Using the CUDA_COMPILE macro
#set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
#set(source_files test_bin.cu)
#CUDA_COMPILE(CUDA_FILES test_bin.cu)
#
#ADD_EXECUTABLE(cuda_compile_example
#  ${CUDA_FILES}
#  ${source_files}
#  main.cc
#  external_dependency.h
#  )
#TARGET_LINK_LIBRARIES(cuda_compile_example
# ${CUDA_LIBRARIES}
# )
#
## Generating PTX files.
## CUDA_COMPILE_PTX(PTX_FILES CUDA_FILES test_bin.cu)
#
## Add a special target to clean nvcc generated files.
#CUDA_BUILD_CLEAN_TARGET()
#

ADD_TEST(CPU-MULTIPLY ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-CPU-MULTIPLY ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_mult.mhd 1 0
)



ADD_TEST(CPU-MULTIPLYC ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-CPU-MULTIPLYC ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_multc.mhd 11 1
)


ADD_TEST(CPU-ABS ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-CPU-ABS ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_abs.mhd 1 1
)


ADD_TEST(CPU-BT ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-CPU-BINARYT ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_bt.mhd 1 100
)


ADD_TEST(CPU-DIVIDE ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-CPU-DIVIDE ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_div.mhd 1 0
)


ADD_TEST(CPU-DIVIDEC ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-CPU-DIVIDEC ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_dividec.mhd 11 1
)



ADD_TEST(CPU-MAXIMUM ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-CPU-MAXIMUM ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead2.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_max.mhd 1 1
)

ADD_TEST(CPU-MINIMUM ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-CPU-MINIMUM ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead2.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_min.mhd 1 1
)


ADD_TEST(CPU-SUBTRACT ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-CPU-SUBTRACT ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead2.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_sub.mhd 1 1
)


ADD_TEST(CPU-SUBTRACTC ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-CPU-SUBTRACTC ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_subc.mhd 11 1
)



ADD_TEST(CPU-ADD ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-CPU-ADD ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_add.mhd 1 1
)



ADD_TEST(CPU-ADDC ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-CPU-ADDC ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_addc.mhd 11 1
)


ADD_TEST(CPU-RESCALE ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-CPU-RESCALE ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_rescale.mhd 1 50 150
)


ADD_TEST(CPU-STATS ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-CPU-STATISTICS ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_stats.mhd 
)


ADD_TEST(CPU-MEAN ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-CPU-MEAN ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_mean_11.mhd 11
)


ADD_TEST(CPU-ERODE ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-CPU-ERODE ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_erode_11.mhd 11
)



ADD_TEST(CPU-DILATE ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-CPU-DILATE ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_dilate_11.mhd 11
)


ADD_TEST(CPU-CLOSE ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-CPU-CLOSE ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_close_11.mhd 11
)



ADD_TEST(CPU-OPEN ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-CPU-OPEN ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_open_11.mhd 11
)



IF (CITK_USE_CUDA)

ADD_TEST(GPU-MULTIPLY ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-MULTIPLY ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_mult.mhd 1 1
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_mult.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_mult.mhd 
)

ADD_TEST(GPU-MULTIPLY2 ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-MULTIPLY ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_mult2.mhd 1 0
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_mult2.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_mult.mhd 
)

ADD_TEST(GPU-MULTIPLYC ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-MULTIPLYC ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_multc.mhd 11 1
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_multc.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_multc.mhd 
)

ADD_TEST(GPU-MULTIPLYC2 ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-MULTIPLYC ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_multc2.mhd 11 0
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_multc2.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_multc.mhd 
)

ADD_TEST(GPU-ABS ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-ABS ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_abs.mhd 1 1
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_abs.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_abs.mhd 
)

ADD_TEST(GPU-ABS2 ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-ABS ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_abs2.mhd 1 0
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_abs2.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_abs.mhd 
)

ADD_TEST(GPU-BT ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-BINARYT ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_bt.mhd 1 100
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_bt.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_bt.mhd 
)

ADD_TEST(GPU-BT2 ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-BINARYT ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_bt2.mhd 0 100
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_bt2.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_bt.mhd 
)

ADD_TEST(GPU-DIVIDE ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-DIVIDE ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_div.mhd 1 1
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_div.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_div.mhd 
)

ADD_TEST(GPU-DIVIDE2 ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-DIVIDE ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_div2.mhd 1 0
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_div2.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_div.mhd 
)

ADD_TEST(GPU-DIVIDEC ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-DIVIDEC ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_dividec.mhd 11 1
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_dividec.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_dividec.mhd 
)

ADD_TEST(GPU-DIVIDEC2 ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-DIVIDEC ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_dividec2.mhd 11 0
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_dividec2.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_dividec.mhd 
)

ADD_TEST(GPU-MAXIMUM ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-MAXIMUM ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead2.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_max.mhd 1 1
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_max.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_max.mhd 
)

ADD_TEST(GPU-MAXIMUM2 ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-MAXIMUM ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead2.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_max2.mhd 1 0
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_max2.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_max.mhd 
)

ADD_TEST(GPU-MINIMUM ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-MINIMUM ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead2.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_min.mhd 1 1
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_min.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_min.mhd 
)

ADD_TEST(GPU-MINIMUM2 ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-MINIMUM ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead2.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_min2.mhd 1 0
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_min2.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_min.mhd 
)

ADD_TEST(GPU-SUBTRACT ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-SUBTRACT ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead2.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_sub.mhd 1 1
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_sub.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_sub.mhd 
)

ADD_TEST(GPU-SUBTRACT2 ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-SUBTRACT ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead2.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_sub2.mhd 1 0
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_sub2.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_sub.mhd 
)

ADD_TEST(GPU-SUBTRACTC ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-SUBTRACTC ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_subc.mhd 11 1
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_subc.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_subc.mhd 
)

ADD_TEST(GPU-SUBTRACTC2 ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-SUBTRACTC ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_subc2.mhd 11 0
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_subc2.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_subc.mhd 
)

ADD_TEST(GPU-ADD ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-ADD ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_add.mhd 1 1
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_add.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_add.mhd 
)

ADD_TEST(GPU-ADD2 ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-ADD ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_add2.mhd 1 0
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_add2.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_add.mhd 
)

ADD_TEST(GPU-ADDC ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-ADDC ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_addc.mhd 11 1
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_addc.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_addc.mhd 
)

ADD_TEST(GPU-ADDC2 ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-ADDC ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_addc2.mhd 11 0
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_addc2.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_addc.mhd 
)

ADD_TEST(GPU-RESCALE ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-RESCALE ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_rescale.mhd 1 50 150
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_rescale.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_rescale.mhd 
)

ADD_TEST(GPU-STATS ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-STATISTICS ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_stats.mhd 
		      #--compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_rescale.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_rescale.mhd 
)

ADD_TEST(GPU-MEAN ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-MEAN ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_mean_11.mhd 11
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_mean_11.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_mean_11.mhd 
)

ADD_TEST(GPU-ERODE ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-ERODE ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_erode_11.mhd 11
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_erode_11.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_erode_11.mhd 
)

ADD_TEST(GPU-DILATE ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-DILATE ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_dilate_11.mhd 11
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_dilate_11.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_dilate_11.mhd 
)

ADD_TEST(GPU-CLOSE ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-CLOSE ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_close_11.mhd 11
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_close_11.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_close_11.mhd 
)

ADD_TEST(GPU-OPEN ${TEST_COMMAND}
		      ${CMAKE_CURRENT_BINARY_DIR}/ITK-GPU-OPEN ${CMAKE_CURRENT_SOURCE_DIR}/images/cthead1.png ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_open_11.mhd 11
		      --compare ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/gpu_open_11.mhd ${CMAKE_CURRENT_SOURCE_DIR}/TestOutputs/cpu_open_11.mhd 
)

ENDIF (CITK_USE_CUDA)
