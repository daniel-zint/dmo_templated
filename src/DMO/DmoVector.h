#pragma once

#include "types.h"

#include <type_traits>
#include <vector>

namespace DMO {

#ifdef _DMO_USE_CUDA
    template<typename T> class DmoVectorGPU {
        std::vector<T> h_;
        DMO::detail::device_vector<T> d_;

      public:
        const auto& h() const { return h_; }
        auto& h() { return h_; }
        const auto& d() const { return d_; }
        auto& d() { return d_; }

        void copyHostToDevice() { d_ = h_; }

        void copyDeviceToHost() { thrust::copy( std::begin( d_ ), std::end( d_ ), std::begin( h_ ) ); }
    };
#endif    // _DMO_USE_CUDA

    template<typename T> class DmoVectorCPU {
        std::vector<T> h_;

      public:
        const auto& h() const { return h_; }
        auto& h() { return h_; }

        void copyHostToDevice(){}
    };

#ifdef _DMO_USE_CUDA
    template<typename T, bool useGPU = true> using DmoVector = typename std::conditional<useGPU, DmoVectorGPU<T>, DmoVectorCPU<T>>::type;
#else
    template<typename T, bool useGPU = true> using DmoVector = DmoVectorCPU<T>;
#endif    // _DMO_USE_CUDA

}    // namespace DMO