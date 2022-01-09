#pragma once

#include "DmoParams.h"
#include "Vertex.h"

#include <cfloat>
#include <cstdint>
#include <stdio.h>
#include <type_traits>

namespace DMO {
    template<typename MetricT>
    void optimizeHierarchical( const int cid, std::vector<float2>& points, const std::vector<int>& coloredVertexIDs, const int cOff,
                               const std::vector<DMO::DmoVertex>& vertices, const std::vector<int>& oneRingVec, const MetricT metric ) {
        const DMO::DmoVertex& v = vertices[coloredVertexIDs[cOff + cid]];

        float q = -FLT_MAX;

        float2 currPos;    // current optimal position
        float2 maxDist;

        float2 oneRing[DMO::MAX_ONE_RING_SIZE];

        // min/max search + loading oneRing
        {
            maxDist.x = -FLT_MAX;
            maxDist.y = -FLT_MAX;

            for( int k = 0; k < v.oneRingSize - 1; ++k ) {
                float2 vo  = points[oneRingVec[v.oneRingID + k]];
                oneRing[k] = vo;

                float2 dist = { std::abs( points[v.idx].x - vo.x ), std::abs( points[v.idx].y - vo.y ) };

                maxDist.x = fmaxf( maxDist.x, dist.x );
                maxDist.y = fmaxf( maxDist.y, dist.y );
            }

            oneRing[v.oneRingSize - 1] = points[oneRingVec[v.oneRingID + v.oneRingSize - 1]];

            currPos = points[v.idx];
        }

        // start depth iteration
        float depth_scale = DMO::GRID_SCALE;
        for( int depth = 0; depth < DMO::DEPTH; ++depth ) {
            float2 aabbMin, aabbMax;    // axis aligned bounding box
            aabbMin.x = currPos.x - depth_scale * maxDist.x;
            aabbMin.y = currPos.y - depth_scale * maxDist.y;
            aabbMax.x = currPos.x + depth_scale * maxDist.x;
            aabbMax.y = currPos.y + depth_scale * maxDist.y;

            float qMax  = -FLT_MAX;
            float2 pMax = { FLT_MAX, FLT_MAX };

            for( int i = 0; i < DMO::NQ; ++i ) {
                for( int j = 0; j < DMO::NQ; ++j ) {
                    float2 p = { DMO::AFFINE_FACTOR * ( i * aabbMin.x + ( DMO::NQ - 1 - i ) * aabbMax.x ),
                                 DMO::AFFINE_FACTOR * ( j * aabbMin.y + ( DMO::NQ - 1 - j ) * aabbMax.y ) };

                    float q = metric.vertexQuality( oneRing, v.oneRingSize, points[v.idx], p );
                    if( q > qMax ) {
                        qMax = q;
                        pMax = p;
                    }
                }
            }

            float qOld = metric.vertexQuality( oneRing, v.oneRingSize, points[v.idx], currPos );
            if( qMax > qOld ) {
                currPos = pMax;
            }

            // rescale candidate grid to the size of two cells
            depth_scale *= 2 * DMO::AFFINE_FACTOR;
        }

        points[v.idx] = currPos;
    }

    template<typename MetricT> void dmoCPU( std::vector<float2>& points, DmoMesh<false>& dmoMesh1_, const MetricT& metric ) {
        for( auto colorIt = dmoMesh1_.colorOffset_.begin(); colorIt != dmoMesh1_.colorOffset_.end() - 1; ++colorIt ) {
            const int nBlocks = *( colorIt + 1 ) - *colorIt;
#pragma omp parallel for
            for( int i = 0; i < nBlocks; ++i ) {
                optimizeHierarchical( i, points, dmoMesh1_.coloredVertexIDs.h(), *colorIt, dmoMesh1_.vertices.h(), dmoMesh1_.oneRingVec.h(), metric );
            }
        }
    }
}    // namespace DMO
#undef REGISTER_METRIC
#define REGISTER_METRIC( metric ) template void DMO::dmoCPU<>( std::vector<float2>&, DMO::DmoMesh<false>&, const metric& );

#include "../DMO/Metrics.h"