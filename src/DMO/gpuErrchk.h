#pragma once

namespace DMO {
    // Cuda error output
#define gpuErrchk( ans )                                                                                                                             \
    { gpuAssert( ( ans ), __FILE__, __LINE__ ); }
    inline void gpuAssert( cudaError_t code, const char* file, int line, bool abort = true ) {
        if( code != cudaSuccess ) {
            // fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            fprintf( stderr, "GPUassert: %s. Line %d\n", cudaGetErrorString( code ), line );
            if( abort )
                exit( code );
        }
    }
}    // namespace DMO