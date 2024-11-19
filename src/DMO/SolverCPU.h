#pragma once

#include "DmoMesh.h"
#include "DmoParams.h"
#include "Metrics.h"
#include "Vertex.h"
#include "types.h"

#include <type_traits>
#include <vector>

namespace DMO {
    namespace detail {
        // CPU kernel
        template<typename MetricT> void dmoCPU( std::vector<float2>& points, DmoMesh<false>& dmoMesh1_, const MetricT& metric );
    }    // namespace detail

    template<typename MeshT, typename MetricT, typename Metric2T = DMO::Metrics::NoMetric> class SolverCPU {
        MeshT& mesh_;
        MetricT* metric1_;
        Metric2T* metric2_;

        std::vector<float2> points_;

        DmoMesh<false>* dmoMesh1_;
        DmoMesh<false>* dmoMesh2_;

      public:
        SolverCPU<MeshT, MetricT, Metric2T>( MeshT& mesh, MetricT* metric1, DmoMesh<false>* dmoMesh1, Metric2T* metric2 = nullptr,
                                             DmoMesh<false>* dmoMesh2 = nullptr );

        void solve( int nIterations = 100 );
    };

    /*
     *		========================
     *		Implementation of Solver
     *		========================
     */

    template<typename MeshT, typename MetricT, typename Metric2T>
    SolverCPU<MeshT, MetricT, Metric2T>::SolverCPU( MeshT& mesh, MetricT* metric1, DmoMesh<false>* dmoMesh1, Metric2T* metric2,
                                                    DmoMesh<false>* dmoMesh2 )
        : mesh_( mesh ), metric1_( metric1 ), metric2_( metric2 ), dmoMesh1_( dmoMesh1 ), dmoMesh2_( dmoMesh2 ) {
        // copy points
        points_.resize( mesh_.n_vertices() );
        for( auto vh: mesh_.vertices() ) {
            auto p            = mesh_.point( vh );
            points_[vh.idx()] = { p[0], p[1] };
        }
    }

    template<typename MeshT, typename MetricT, typename Metric2T> void SolverCPU<MeshT, MetricT, Metric2T>::solve( int nIterations ) {
        for( int i = 0; i < nIterations; ++i ) {
            detail::dmoCPU<MetricT>( points_, *dmoMesh1_, *metric1_ );
            if constexpr( !std::is_same<Metric2T, DMO::Metrics::NoMetric>::value ) {
                detail::dmoCPU<Metric2T>( points_, *dmoMesh2_, *metric2_ );
            }
        }

        for( const auto& vh: mesh_.vertices() ) {
            int idx          = vh.idx();
            TriMesh::Point p = { points_[idx].x, points_[idx].y, 0.f };
            mesh_.set_point( vh, p );
        }
    }

}    // namespace DMO

#include "SolverCPU_impl.h"