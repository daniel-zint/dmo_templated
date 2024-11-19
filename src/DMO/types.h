#pragma once

#include <cmath>
#include <type_traits>

#ifdef _DMO_USE_CUDA
#    include "cuda_runtime.h"
#    include "device_launch_parameters.h"
#    include "thrust/device_vector.h"
#    include "vector_types.h"

namespace DMO {
    namespace detail {
        template<typename T, typename = std::enable_if_t<std::is_trivial_v<T>>> using device_vector = thrust::device_vector<T>;
    }
}    // namespace DMO

#else
struct float2 {
    float x;
    float y;
};
#endif    // _DMO_USE_CUDA