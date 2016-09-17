cmake_minimum_required(VERSION 2.4)
project(sgm)
find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_52,code=sm_52 -lineinfo
    )

cuda_add_executable(
    sgm
    main.cu median_filter.cu hamming_cost.cu disparity_method.cu debug.cu costs.cu)

target_link_libraries( sgm ${OpenCV_LIBS} )