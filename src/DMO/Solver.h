#pragma once

#include "DmoMesh.h"
#include "DmoParams.h"
#include "Metrics.h"
#include "Vertex.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gpuErrchk.h"
#include "math_constants.h"

#include <type_traits>

namespace DMO {
    namespace detail {
        // CUDA kernel
        template<typename MetricT> void dmoGPU( float2* const points_d, DmoMesh<true>& dmoMesh1_, const MetricT& metric );
    }    // namespace detail

    template<typename MeshT, typename MetricT, typename Metric2T = DMO::Metrics::NoMetric> class Solver {
        MeshT& mesh_;
        MetricT* metric1_;
        Metric2T* metric2_;

        DmoVector<float2> points_;

        DmoMesh<true>* dmoMesh1_;
        DmoMesh<true>* dmoMesh2_;

      public:
        Solver<MeshT, MetricT, Metric2T>( MeshT& mesh, MetricT* metric1, DmoMesh<true>* dmoMesh1, Metric2T* metric2 = nullptr,
                                          DmoMesh<true>* dmoMesh2 = nullptr );

        void solve( int nIterations = 100 );
    };

    /*
     *		========================
     *		Implementation of Solver
     *		========================
     */

    template<typename MeshT, typename MetricT, typename Metric2T>
    Solver<MeshT, MetricT, Metric2T>::Solver( MeshT& mesh, MetricT* metric1, DmoMesh<true>* dmoMesh1, Metric2T* metric2, DmoMesh<true>* dmoMesh2 )
        : mesh_( mesh ), metric1_( metric1 ), metric2_( metric2 ), dmoMesh1_( dmoMesh1 ), dmoMesh2_( dmoMesh2 ) {
        // copy points
        points_.h().resize( mesh_.n_vertices() );
        for( auto vh: mesh_.vertices() ) {
            auto p            = mesh_.point( vh );
            points_.h()[vh.idx()] = { p[0], p[1] };
        }
        points_.copyHostToDevice();

        dmoMesh1_->copyHostToDevice();
        if constexpr( !std::is_same<Metric2T, Metrics::NoMetric>::value ) {
            dmoMesh2_->copyHostToDevice();
        }

        gpuErrchk( cudaDeviceSynchronize() );
    }

    template<typename MeshT, typename MetricT, typename Metric2T> void Solver<MeshT, MetricT, Metric2T>::solve( int nIterations ) {
        for( int i = 0; i < nIterations; ++i ) {
            detail::dmoGPU<MetricT>( points_.d().data().get(), *dmoMesh1_, *metric1_ );
            if constexpr( !std::is_same<Metric2T, Metrics::NoMetric>::value ) {
                detail::dmoGPU<Metric2T>( points_.d().data().get(), *dmoMesh2_, *metric2_ );
            }
        }
        points_.copyDeviceToHost();

        for( const auto& vh: mesh_.vertices() ) {
            int idx          = vh.idx();
            TriMesh::Point p = { points_.h()[idx].x, points_.h()[idx].y, 0.f };
            mesh_.set_point( vh, p );
        }
    }

}    // namespace DMO