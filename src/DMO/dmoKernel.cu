#include "DmoMesh.h"
#include "DmoParams.h"
#include "Vertex.h"
// CUDA stuff
#include "device_launch_parameters.h"
#include "gpuErrchk.h"

#include <cfloat>
#include <cstdint>
#include <stdio.h>
#include <type_traits>

namespace DMO {
    namespace detail {
        template<typename T, typename ShuffleType = int> __device__ inline T shfl_xor( T var, unsigned int srcLane, int width = 32 ) {
            static_assert( sizeof( T ) % sizeof( ShuffleType ) == 0, "Cannot shuffle this type." );
            ShuffleType* a = reinterpret_cast<ShuffleType*>( &var );
            for( int i = 0; i < sizeof( T ) / sizeof( ShuffleType ); ++i ) {
                a[i] = __shfl_xor_sync( 0xFFFFFFFF, a[i], srcLane, width );
            }
            return var;
        }

        template<typename T, typename OP> __device__ inline T warpReduce( T val, OP op ) {
            static_assert( sizeof( T ) <= 8, "Only 8 byte reductions are supportet." );
            for( int offset = warpSize >> 1; offset > 0; offset >>= 1 ) {
                auto v = shfl_xor( val, offset );
                val    = op( val, v );
            }
            return val;
        }

        template<typename MetricT>
        __global__ void optimizeHierarchical( float2* const points, const int* coloredVertexIDs, const int cOff, const DmoVertex* vertices,
                                              const int* oneRingVec, const MetricT metric ) {
            const uint2 idx1 = { threadIdx.x / NQ, threadIdx.x % NQ };
            const uint2 idx2 = { ( threadIdx.x + NQ * NQ / 2 ) / NQ, ( threadIdx.x + NQ * NQ / 2 ) % NQ };

            const DmoVertex& v = vertices[coloredVertexIDs[cOff + blockIdx.x]];

            float q = -FLT_MAX;

            float2 currPos = points[v.idx];    // current optimal position

            __shared__ float2 oneRing[MAX_ONE_RING_SIZE];
            float2 maxDist;

            // min/max search + loading oneRing
            float2 vo;
            if( threadIdx.x < v.oneRingSize ) {
                vo                   = points[oneRingVec[v.oneRingID + threadIdx.x]];
                oneRing[threadIdx.x] = vo;
            } else {
                vo = currPos;
            }
            __syncwarp();

            maxDist.x = warpReduce( std::abs( currPos.x - vo.x ), []( auto d1, auto d2 ) { return d1 > d2 ? d1 : d2; } );
            maxDist.y = warpReduce( std::abs( currPos.y - vo.y ), []( auto d1, auto d2 ) { return d1 > d2 ? d1 : d2; } );

            thrust::pair<float, int> opt;
            opt.first  = metric.vertexQuality( oneRing, v.oneRingSize, points[v.idx], currPos );
            opt.second = NQ * NQ;

            // start depth iteration
            float depth_scale = GRID_SCALE;
            for( int depth = 0; depth < DEPTH; ++depth ) {
                float2 aabbMin, aabbMax;    // axis aligned bounding box
                aabbMin.x = currPos.x - depth_scale * maxDist.x;
                aabbMin.y = currPos.y - depth_scale * maxDist.y;
                aabbMax.x = currPos.x + depth_scale * maxDist.x;
                aabbMax.y = currPos.y + depth_scale * maxDist.y;

                float2 p1 = { AFFINE_FACTOR * ( idx1.x * aabbMin.x + ( NQ - 1 - idx1.x ) * aabbMax.x ),
                              AFFINE_FACTOR * ( idx1.y * aabbMin.y + ( NQ - 1 - idx1.y ) * aabbMax.y ) };
                float2 p2 = { AFFINE_FACTOR * ( idx2.x * aabbMin.x + ( NQ - 1 - idx2.x ) * aabbMax.x ),
                              AFFINE_FACTOR * ( idx2.y * aabbMin.y + ( NQ - 1 - idx2.y ) * aabbMax.y ) };

                float q1 = metric.vertexQuality( oneRing, v.oneRingSize, points[v.idx], p1 );
                float q2 = metric.vertexQuality( oneRing, v.oneRingSize, points[v.idx], p2 );
                float2 p;
                if( q1 > q2 ) {
                    q = q1;
                    p = p1;
                } else {
                    q = q2;
                    p = p2;
                }

                thrust::pair<float, int> data( q, idx1.x * NQ + idx1.y );
                data = warpReduce( data, []( auto q1, auto q2 ) { return q1.first > q2.first ? q1 : q2; } );

                opt = data;

                // __syncwarp();

                float qOld = metric.vertexQuality( oneRing, v.oneRingSize, points[v.idx], currPos );

                // keep old position if new quality is not better
                if( q <= qOld ) {
                    p = currPos;
                }
                __syncwarp();
                currPos.x = __shfl_sync( 0xFFFFFFFF, p.x, opt.second );
                currPos.y = __shfl_sync( 0xFFFFFFFF, p.y, opt.second );

                // rescale candidate grid to the size of two cells
                depth_scale *= 2 * AFFINE_FACTOR;
            }

            // set new position if it is better than the old one
            float qOld = metric.vertexQuality( oneRing, v.oneRingSize, points[v.idx], points[v.idx] );
            if( idx1.x * NQ + idx1.y == opt.second && qOld < q ) {
                points[v.idx] = currPos;
            }
        }

        template<typename MetricT> void dmoGPU( float2* const points_d, DmoMesh<true>& dmoMesh1_, const MetricT& metric ) {
            for( auto colorIt = dmoMesh1_.colorOffset_.begin(); colorIt != dmoMesh1_.colorOffset_.end() - 1; ++colorIt ) {
                const int nBlocks = *( colorIt + 1 ) - *colorIt;
                optimizeHierarchical<<<nBlocks, N_THREADS>>>( points_d, dmoMesh1_.coloredVertexIDs.d().data().get(), *colorIt,
                                                              dmoMesh1_.vertices.d().data().get(), dmoMesh1_.oneRingVec.d().data().get(), metric );
            }
        }
    }    // namespace detail
}    // namespace DMO

#define REGISTER_METRIC( metric ) template void DMO::detail::dmoGPU<>( float2* const, DMO::DmoMesh<true>&, const metric& )

#include "Metrics.h"
