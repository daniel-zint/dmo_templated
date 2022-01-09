#pragma once

namespace DMO {
    // Size of Quality Mesh
    constexpr unsigned int NQ = 8;
    // number of refinement steps within DMO
    constexpr int DEPTH = 3;
    // the maximal number of allowed vertices on the one-ring neighborhood
    constexpr int MAX_ONE_RING_SIZE = 16;
    // scaling factor between DMO grid and the bounding box of the one-ring
    constexpr float GRID_SCALE = 0.5f;
    // affine factor for computing DMO grid node positions
    constexpr float AFFINE_FACTOR = 1.f / (float)( NQ - 1 );

    constexpr int N_THREADS = NQ * NQ / 2;
}    // namespace DMO