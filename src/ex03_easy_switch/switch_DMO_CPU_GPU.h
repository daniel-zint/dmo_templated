#pragma once

#if defined(_DMO_USE_CUDA) && !defined(_FORCE_CPU_USAGE)
#include "DMO/Solver.h"
    namespace utility {
    constexpr bool use_GPU = true;
    template<typename MeshT, typename MetricT, typename Metric2T = DMO::Metrics::NoMetric>
    class MyDMO : public DMO::Solver<MeshT, MetricT, Metric2T> {
    public:
        using DMO::Solver<MeshT, MetricT, Metric2T>::Solver;
    };

    template<typename MeshT, typename MetricT, typename Metric2T = DMO::Metrics::NoMetric>
    MyDMO(  MeshT & mesh,
            MetricT * metric1,
            DMO::DmoMesh<true>* dmoMesh1,
            Metric2T* metric2 = nullptr,
            DMO::DmoMesh<true>* dmoMesh2 = nullptr) -> MyDMO<MeshT, MetricT, Metric2T>;
    }
#else
#include "DMO/SolverCPU.h"
    namespace utility {
    constexpr bool use_GPU = false;

    template<typename MeshT, typename MetricT, typename Metric2T = DMO::Metrics::NoMetric>
    class MyDMO : public DMO::SolverCPU<MeshT, MetricT, Metric2T> {
    public:
        using DMO::SolverCPU<MeshT, MetricT, Metric2T>::SolverCPU;
    };

    template<typename MeshT, typename MetricT, typename Metric2T = DMO::Metrics::NoMetric>
    MyDMO(  MeshT & mesh,
            MetricT * metric1,
            DMO::DmoMesh<false>* dmoMesh1,
            Metric2T* metric2 = nullptr,
            DMO::DmoMesh<false>* dmoMesh2 = nullptr) -> MyDMO<MeshT, MetricT, Metric2T>;
}
#endif
