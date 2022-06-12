/*
 * DMO Example 04 - User Metrics on either CPU or GPU chosen at compile time, selecting the one available
 *
 * The user may force CPU usage even if GPU is available by defining _FORCE_CPU_USAGE.
 * 
 * The user can define own metrics, see UserMetrics.h. The metrics must be registered.
 *
 * A constexpr bool variable utility::use_GPU is provided in combination with an unified
 * name for accessing DMO::Solver and DMO::SolverCPU by calling utility::MyDMO.
 *
 * !!! All metric-files must be included after include of switch_DMO_CPU_GPU.h !!!
 */

#include "baseDirectory.h"

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <experimental/filesystem>
#include <iostream>
typedef OpenMesh::TriMesh_ArrayKernelT<> TriMesh;


#include "switch_DMO_CPU_GPU.h"

#include "UserMetrics.h"

namespace fs = std::experimental::filesystem;

int main( int argc, char* argv[] ) {
    fs::current_path( BASE_DIRECTORY );

    fs::path meshPath = "examples/transition4to1.off";

    TriMesh mesh;
    OpenMesh::IO::read_mesh( mesh, meshPath.string() );

    MyMeanRatioTriangle myMetric;

    DMO::DmoMesh<utility::use_GPU> dmoMeshAll = DMO::DmoMesh<utility::use_GPU>::create<TriMesh, DMO::Set::All>( mesh );

    std::cout << "Start solver" << std::endl;

    utility::MyDMO( mesh, &myMetric, &dmoMeshAll ).solve( 100 );
   
    std::cout << "Finish solver" << std::endl;

    /* 
     * DMO can take two sets of vertices with different metrics. In this case
     * it is not really reasonable to use two sets. It's basically just to show
     * that something like that is possible. DMO toggles between the two sets
     * and metrics performing the number of iterations on both.
     *
     * Depending on the Hardware it is executed on the smoothing order will be 
     * different because the coloring is computed for each vertex set 
     * individually.
     */
    //DMO::DmoMesh<utility::use_GPU> dmoMeshInner = DMO::DmoMesh<utility::use_GPU>::create<TriMesh, DMO::Set::Inner>( mesh );
    //DMO::DmoMesh<utility::use_GPU> dmoMeshBound = DMO::DmoMesh<utility::use_GPU>::create<TriMesh, DMO::Set::Boundary>( mesh );
    //utility::MyDMO( mesh, &myMetric, &dmoMeshInner, &myMetric, &dmoMeshBound ).solve( 100 );

    OpenMesh::IO::write_mesh( mesh, "examples/out.off" );
}