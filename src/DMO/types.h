#pragma once

#include <cmath>

#ifdef _DMO_USE_CUDA
#    include "cuda_runtime.h"
#    include "device_launch_parameters.h"
#    include "thrust/device_vector.h"
#    include "thrust/host_vector.h"
#    include "vector_types.h"
#else
struct float2 {
    float x;
    float y;
};
#endif    // _DMO_USE_CUDA