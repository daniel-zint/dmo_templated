#pragma once
#include "DMO/Metrics.h"

class MyMeanRatioTriangle {
    static constexpr int elementSize = 3;

    __device__ float triangleQuality( const float2 p[3] ) const {
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

  public:
    __device__ __forceinline__ float vertexQuality( const float2* oneRing, const int oneRingSize, const float2& pCurr, const float2& p ) const {
        float q = FLT_MAX;

        for( int k = 0; k < oneRingSize - 1; ++k ) {
            float2 t[3] = { { p.x, p.y }, { oneRing[k].x, oneRing[k].y }, { oneRing[k + 1].x, oneRing[k + 1].y } };
            q           = fminf( q, triangleQuality( t ) );
        }

        return q;
    }
};

REGISTER_METRIC( MyMeanRatioTriangle );