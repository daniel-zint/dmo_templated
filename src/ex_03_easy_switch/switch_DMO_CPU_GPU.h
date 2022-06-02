#pragma once

#if defined(_DMO_USE_CUD) && !defined(_FORCE_CPU_USAGE)
#include "DMO/Solver.h"
    namespace utility {
    constexpr bool use_GPU = true;
    template<typename MeshT, typename MetricT, typename Metric2T = DMO::Metrics::NoMetric>
    using Solver = DMO::Solver<MeshT,MetricT,Metric2T>;
    }
#else
#include "DMO/SolverCPU.h"
    namespace utility {
    constexpr bool use_GPU = false;
    template<typename MeshT, typename MetricT, typename Metric2T = DMO::Metrics::NoMetric>
    using Solver = DMO::SolverCPU<MeshT,MetricT,Metric2T>;
}
#endif
