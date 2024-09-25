#pragma once

/*
 * Depending on the available resources and the explicit choice of compute location makes the appropriate backend solver for DMO
 * accessable through utility::MyDMOSolver, with standard template argument deduction, leading to nice to read syntax.
 */

#include "DMO/SolverCPU.h"

#if defined( _DMO_USE_CUDA ) && !defined( _FORCE_CPU_USAGE )
#    include "DMO/Solver.h"
#    define IMPL_SWITCH_DMO_USE_GPU true
#else
namespace DMO {
    template<typename... T> using Solver = void;
}
#    define IMPL_SWITCH_DMO_USE_GPU false
#endif

#include <type_traits>

namespace utility {
    constexpr bool use_GPU = IMPL_SWITCH_DMO_USE_GPU;

    namespace {
        // make the to be derived class from avaiable via an alias
        template<typename... T> using ImplSolverBase = std::conditional_t<use_GPU, DMO::Solver<T...>, DMO::SolverCPU<T...>>;
    }    // namespace

    // dervies the to be used class while inheriting the constructors and making them available
    template<typename MeshT, typename MetricT, typename Metric2T = DMO::Metrics::NoMetric>
    class MyDMOSolver : public ImplSolverBase<MeshT, MetricT, Metric2T> {
      public:
        using ImplSolverBase<MeshT, MetricT, Metric2T>::ImplSolverBase;
    };

    // explicit deduction guide for easy use
    template<typename MeshT, typename MetricT, typename Metric2T = DMO::Metrics::NoMetric>
    MyDMOSolver( MeshT& mesh, MetricT* metric1, DMO::DmoMesh<use_GPU>* dmoMesh1, Metric2T* metric2 = nullptr,
                 DMO::DmoMesh<use_GPU>* dmoMesh2 = nullptr ) -> MyDMOSolver<MeshT, MetricT, Metric2T>;

}    // namespace utility

#undef IMPL_SWITCH_DMO_USE_GPU