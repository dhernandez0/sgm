cmake_minimum_required(VERSION 2.4)
project(sgm)
find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -lineinfo
    -gencode=arch=compute_30,code=sm_30
    -gencode=arch=compute_35,code=sm_35
    -gencode=arch=compute_50,code=sm_50
    -gencode=arch=compute_52,code=sm_52
    -gencode=arch=compute_52,code=compute_52
    )

cuda_add_executable(
    sgm
    main.cu median_filter.cu hamming_cost.cu disparity_method.cu debug.cu costs.cu)

target_link_libraries( sgm ${OpenCV_LIBS} )
