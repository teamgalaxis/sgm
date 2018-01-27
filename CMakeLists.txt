#    This file is part of sgm. (https://github.com/dhernandez0/sgm).
#
#    Copyright (c) 2016 Daniel Hernandez Juarez.

#    sgm is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    sgm is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with sgm.  If not, see <http://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 2.4)
project(sgm)
find_package( OpenCV REQUIRED )
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-8.0/)
find_package( CUDA 8.0 REQUIRED EXACT)

include_directories(include ${INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    #-O3 -lineinfo
    # Nvidia Jetson TX1
    #-gencode=arch=compute_53,code=sm_53;
    # The beast in the office
    #-gencode=arch=compute_61,code=sm_61;
    # Nvidia Jetson TX2
    -gencode arch=compute_62,code=sm_62;
	-gencode arch=compute_62,code=compute_62 
    -std=c++11
)

cuda_add_executable(
    sgm
	main.cu median_filter.cu hamming_cost.cu disparity_method.cu debug.cu costs.cu)
cuda_add_library(
  sgm_api
        api.cu median_filter.cu hamming_cost.cu debug.cu costs.cu
  SHARED)

target_link_libraries( sgm ${OpenCV_LIBS} )
target_link_libraries( sgm_api ${OpenCV_LIBS} ${CUDA_LIBRARIES})

add_dependencies(sgm_api sgm)
