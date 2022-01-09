#pragma once

#include "DmoParams.h"
#include "types.h"

#ifndef _DMO_USE_CUDA
#    define __device__
#    define __forceinline__ inline
#endif    // DMO_USE_CUDA

namespace DMO {
    namespace Metrics {
        struct NoMetric {
            __device__ __forceinline__ float vertexQuality( const float2* oneRing, const int oneRingSize, const float2& pCurr,
                                                            const float2& p ) const {
                return -FLT_MAX;
            }
        };

        namespace Tri {
            struct MeanRatio {
                static constexpr int elementSize = 3;

                __device__ __forceinline__ float elementQuality( const float2 p[3] ) const {
                    float2 e[3];
                    float e_length_squared[3];

                    for( int i = 0; i < 3; ++i ) {
                        int j  = ( i + 1 ) % 3;
                        e[i].x = p[j].x - p[i].x;
                        e[i].y = p[j].y - p[i].y;

                        e_length_squared[i] = e[i].x * e[i].x + e[i].y * e[i].y;
                    }

                    float l    = e_length_squared[0] + e_length_squared[1] + e_length_squared[2];
                    float area = e[0].x * e[1].y - e[0].y * e[1].x;

                    if( area < 0 )
                        return area;
                    else
                        return 2.f * sqrtf( 3.f ) * area / l;
                }

                __device__ __forceinline__ float vertexQuality( const float2* oneRing, const int oneRingSize, const float2& pCurr,
                                                                const float2& p ) const {
                    float q = FLT_MAX;

                    for( int k = 0; k < oneRingSize - 1; ++k ) {
                        float2 t[3] = { { p.x, p.y }, { oneRing[k].x, oneRing[k].y }, { oneRing[k + 1].x, oneRing[k + 1].y } };
                        q           = fminf( q, elementQuality( t ) );
                    }

                    return q;
                }
            };

            struct MinAngle {
                static constexpr int elementSize = 3;

                __device__ __forceinline__ float calcAngle( const float2& p1, const float2& p2, const float2& p3 ) const {
                    const float dot = ( p1.x - p2.x ) * ( p3.x - p2.x ) + ( p1.y - p2.y ) * ( p3.y - p2.y );
                    const float det = ( p1.x - p2.x ) * ( p3.y - p2.y ) - ( p1.y - p2.y ) * ( p3.x - p2.x );
                    return fabsf( atan2f( det, dot ) );
                }

                __device__ __forceinline__ float elementQuality( const float2 p[3] ) const {
                    float2 e[3];

                    for( int i = 0; i < 3; ++i ) {
                        int j  = ( i + 1 ) % 3;
                        e[i].x = p[j].x - p[i].x;
                        e[i].y = p[j].y - p[i].y;
                    }

                    float a = calcAngle( p[0], p[1], p[2] );
                    float b = calcAngle( p[1], p[2], p[0] );
                    float c = calcAngle( p[2], p[0], p[1] );

                    float min_angle = fminf( a, b );
                    min_angle       = fminf( min_angle, c );

                    float area = e[0].x * e[1].y - e[0].y * e[1].x;

                    float q = 3.f * min_angle / (float)M_PI;

                    // if triangle is flipped, make value negative
                    if( area < 0 )
                        return -q;
                    else
                        return q;
                }

                __device__ __forceinline__ float vertexQuality( const float2* oneRing, const int oneRingSize, const float2& pCurr,
                                                                const float2& p ) const {
                    float q = FLT_MAX;

                    for( int k = 0; k < oneRingSize - 1; ++k ) {
                        float2 t[3] = { { p.x, p.y }, { oneRing[k].x, oneRing[k].y }, { oneRing[k + 1].x, oneRing[k + 1].y } };
                        q           = fminf( q, elementQuality( t ) );
                    }

                    return q;
                }
            };

            struct RadiusRatio {
                static constexpr int elementSize = 3;

                __device__ __forceinline__ float elementQuality( const float2 p[3] ) const {
                    float2 e[3];
                    float e_length[3];

                    for( int i = 0; i < 3; ++i ) {
                        int j  = ( i + 1 ) % 3;
                        e[i].x = p[j].x - p[i].x;
                        e[i].y = p[j].y - p[i].y;

                        e_length[i] = sqrtf( e[i].x * e[i].x + e[i].y * e[i].y );
                    }

                    float area = 0.5f * ( e[0].x * e[1].y - e[0].y * e[1].x );
                    float s    = 0.5f * ( e_length[0] + e_length[1] + e_length[2] );
                    float ri   = area / s;
                    float ro   = 0.25f * e_length[0] * e_length[1] * e_length[2] / area;

                    float criterion = 2.f * ri / ro;

                    if( criterion > 1 ) {    // might happen due to numerical errors
                        return 1;
                    }

                    return criterion;
                }

                __device__ __forceinline__ float vertexQuality( const float2* oneRing, const int oneRingSize, const float2& pCurr,
                                                                const float2& p ) const {
                    float q = FLT_MAX;

                    for( int k = 0; k < oneRingSize - 1; ++k ) {
                        float2 t[3] = { { p.x, p.y }, { oneRing[k].x, oneRing[k].y }, { oneRing[k + 1].x, oneRing[k + 1].y } };
                        q           = fminf( q, elementQuality( t ) );
                    }

                    return q;
                }
            };

            struct LaplaceDist {
                static constexpr int elementSize = 3;

                __device__ __forceinline__ float vertexQuality( const float2* oneRing, const int oneRingSize, const float2& pCurr,
                                                                const float2& p ) const {
                    float2 pOpt = { 0, 0 };
                    for( int k = 0; k < oneRingSize - 1; ++k ) {
                        pOpt.x += oneRing[k].x;
                        pOpt.y += oneRing[k].y;
                    }
                    pOpt.x /= ( oneRingSize - 1 );
                    pOpt.y /= ( oneRingSize - 1 );

                    float2 d = { pOpt.x - p.x, pOpt.y - p.y };

                    float q = 1.f / ( d.x * d.x + d.y * d.y + 0.0001f );

                    return q;
                }
            };
        }    // namespace Tri

        namespace Quad {
            struct Shape {
                static constexpr int elementSize = 4;

                __device__ __forceinline__ float elementQuality( const float2 p[4] ) const {
                    float2 e[4];
                    float e_length_squared[4];

                    for( int i = 0; i < 4; ++i ) {
                        int j  = ( i + 1 ) % 4;
                        e[i].x = p[j].x - p[i].x;
                        e[i].y = p[j].y - p[i].y;

                        e_length_squared[i] = e[i].x * e[i].x + e[i].y * e[i].y;
                    }

                    float det0 = e[0].y * e[3].x - e[0].x * e[3].y;
                    float det1 = e[1].y * e[0].x - e[1].x * e[0].y;
                    float det3 = e[3].y * e[2].x - e[3].x * e[2].y;

                    float det = fminf( det0, det1 );
                    det       = fminf( det, det3 );

                    if( det < 0 )
                        return det;

                    float c0 = 2 * det0 / ( e_length_squared[0] + e_length_squared[3] );
                    float c1 = 2 * det1 / ( e_length_squared[1] + e_length_squared[0] );
                    float c3 = 2 * det3 / ( e_length_squared[3] + e_length_squared[2] );

                    float c = fminf( c0, c1 );
                    c       = fminf( c, c3 );

                    return c;
                }

                __device__ __forceinline__ float vertexQuality( const float2* oneRing, const int oneRingSize, const float2& pCurr,
                                                                const float2& p ) const {
                    float q = FLT_MAX;

                    for( int k = 0; k < oneRingSize - 1; k += 2 ) {
                        float2 t[4] = { { p.x, p.y },
                                        { oneRing[k].x, oneRing[k].y },
                                        { oneRing[k + 1].x, oneRing[k + 1].y },
                                        { oneRing[k + 2].x, oneRing[k + 2].y } };
                        q           = fminf( q, elementQuality( t ) );
                    }

                    return q;
                }
            };

            struct MinAngle {
                static constexpr int elementSize = 3;

                __device__ __forceinline__ float calcAngle( const float2& p1, const float2& p2, const float2& p3 ) const {
                    const float dot = ( p1.x - p2.x ) * ( p3.x - p2.x ) + ( p1.y - p2.y ) * ( p3.y - p2.y );
                    const float det = ( p1.x - p2.x ) * ( p3.y - p2.y ) - ( p1.y - p2.y ) * ( p3.x - p2.x );
                    return fabsf( atan2f( det, dot ) );
                }

                __device__ __forceinline__ float elementQuality( const float2 p[4] ) const {
                    float2 e[4];
                    float e_length[4];

                    for( int i = 0; i < 4; ++i ) {
                        int j       = ( i + 1 ) % 4;
                        e[i].x      = p[j].x - p[i].x;
                        e[i].y      = p[j].y - p[i].y;
                        e_length[i] = sqrtf( e[i].x * e[i].x + e[i].y * e[i].y );
                    }

                    float a = (float)M_PI - acosf( ( e[0].x * e[1].x + e[0].y * e[1].y ) / ( e_length[0] * e_length[1] ) );
                    float b = (float)M_PI - acosf( ( e[1].x * e[2].x + e[1].y * e[2].y ) / ( e_length[1] * e_length[2] ) );
                    float c = (float)M_PI - acosf( ( e[2].x * e[3].x + e[2].y * e[3].y ) / ( e_length[2] * e_length[3] ) );
                    float d = (float)M_PI - acosf( ( e[3].x * e[0].x + e[3].y * e[0].y ) / ( e_length[3] * e_length[0] ) );

                    float min_angle = fminf( a, b );
                    min_angle       = fminf( min_angle, c );
                    min_angle       = fminf( min_angle, d );

                    return 2.f * min_angle / (float)M_PI;
                }

                __device__ __forceinline__ float vertexQuality( const float2* oneRing, const int oneRingSize, const float2& pCurr,
                                                                const float2& p ) const {
                    float q = FLT_MAX;

                    for( int k = 0; k < oneRingSize - 1; k += 2 ) {
                        float2 t[4] = { { p.x, p.y },
                                        { oneRing[k].x, oneRing[k].y },
                                        { oneRing[k + 1].x, oneRing[k + 1].y },
                                        { oneRing[k + 2].x, oneRing[k + 2].y } };
                        q           = fminf( q, elementQuality( t ) );
                    }

                    return q;
                }
            };

            struct ScaledJacobian {
                static constexpr int elementSize = 4;

                __device__ __forceinline__ float elementQuality( const float2 p[4] ) const {
                    // get jacobian in every point of the quad
                    // take the smallest one
                    // scale it with the incident edges

                    float dxds_1 = ( p[0].x - p[1].x );    // dx/ds (t = 1)
                    float dxds_2 = ( p[3].x - p[2].x );    // dx/ds (t = -1)

                    float dyds_1 = ( p[0].y - p[1].y );    // dy/ds (t = 1)
                    float dyds_2 = ( p[3].y - p[2].y );    // dy/ds (t = -1)

                    float dxdt_1 = ( p[0].x - p[3].x );    // dx/dt (s = 1)
                    float dxdt_2 = ( p[1].x - p[2].x );    // dx/dt (s = -1)

                    float dydt_1 = ( p[0].y - p[3].y );    // dy/dt (s = 1)
                    float dydt_2 = ( p[1].y - p[2].y );    // dy/dt (s = -1)

                    float j[4];
                    j[0] = dxds_1 * dydt_1 - dxdt_1 * dyds_1;    // t = s = 1		// x1
                    j[1] = dxds_1 * dydt_2 - dxdt_2 * dyds_1;    // t = 1, s = -1	// x2
                    j[2] = dxds_2 * dydt_2 - dxdt_2 * dyds_2;    // t = s = -1		// x3
                    j[3] = dxds_2 * dydt_1 - dxdt_1 * dyds_2;    // t = -1, s = 1	// x4

                    float jMin = FLT_MAX;
                    int ji     = 5;
                    for( int i = 0; i < 4; ++i ) {
                        if( j[i] < jMin ) {
                            jMin = j[i];
                            ji   = i;
                        }
                    }

                    int jil = ( ji + 4 - 1 ) % 4;
                    int jir = ( ji + 1 ) % 4;

                    float lEdge = sqrtf( ( p[jil].x - p[ji].x ) * ( p[jil].x - p[ji].x ) + ( p[jil].y - p[ji].y ) * ( p[jil].y - p[ji].y ) );
                    float rEdge = sqrtf( ( p[jir].x - p[ji].x ) * ( p[jir].x - p[ji].x ) + ( p[jir].y - p[ji].y ) * ( p[jir].y - p[ji].y ) );

                    return j[ji] / ( lEdge * rEdge );
                }

                __device__ __forceinline__ float vertexQuality( const float2* oneRing, const int oneRingSize, const float2& pCurr,
                                                                const float2& p ) const {
                    float q = FLT_MAX;

                    for( int k = 0; k < oneRingSize - 1; k += 2 ) {
                        float2 t[4] = { { p.x, p.y },
                                        { oneRing[k].x, oneRing[k].y },
                                        { oneRing[k + 1].x, oneRing[k + 1].y },
                                        { oneRing[k + 2].x, oneRing[k + 2].y } };
                        q           = fminf( q, elementQuality( t ) );
                    }

                    return q;
                }
            };

            struct LaplaceDist {
                static constexpr int elementSize = 4;

                __device__ __forceinline__ float vertexQuality( const float2* oneRing, const int oneRingSize, const float2& pCurr,
                                                                const float2& p ) const {
                    float2 pOpt = { 0, 0 };
                    for( int k = 0; k < oneRingSize - 1; k += 2 ) {
                        pOpt.x += oneRing[k].x;
                        pOpt.y += oneRing[k].y;
                    }
                    pOpt.x /= 0.5f * ( oneRingSize - 1 );
                    pOpt.y /= 0.5f * ( oneRingSize - 1 );

                    float2 d = { pOpt.x - p.x, pOpt.y - p.y };

                    float q = 1.f / ( d.x * d.x + d.y * d.y + 0.0001f );

                    return q;
                }
            };
        }    // namespace Quad

    }    // namespace Metrics
}    // namespace DMO

#ifndef REGISTER_METRIC
#    define REGISTER_METRIC( metric )
#endif

REGISTER_METRIC( DMO::Metrics::NoMetric );
REGISTER_METRIC( DMO::Metrics::Tri::MeanRatio );
REGISTER_METRIC( DMO::Metrics::Tri::MinAngle );
REGISTER_METRIC( DMO::Metrics::Tri::RadiusRatio );
REGISTER_METRIC( DMO::Metrics::Tri::LaplaceDist );
REGISTER_METRIC( DMO::Metrics::Quad::Shape );
REGISTER_METRIC( DMO::Metrics::Quad::MinAngle );
REGISTER_METRIC( DMO::Metrics::Quad::ScaledJacobian );
REGISTER_METRIC( DMO::Metrics::Quad::LaplaceDist );