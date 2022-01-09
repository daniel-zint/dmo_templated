/*
 * DMO Example 02 - User Metrics
 *
 * The user can define own metrics, see UserMetrics.h. The metrics must be registered.
 *
 * !!! All metric-files must be included in registerDMO.cu !!!
 */

#include "baseDirectory.h"

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <experimental/filesystem>
#include <iostream>
typedef OpenMesh::TriMesh_ArrayKernelT<> TriMesh;

#include "DMO/Solver.h"
#include "UserMetrics.h"

namespace fs = std::experimental::filesystem;

int main( int argc, char* argv[] ) {
    fs::current_path( BASE_DIRECTORY );

    fs::path meshPath = "examples/transition4to1.off";

    TriMesh mesh;
    OpenMesh::IO::read_mesh( mesh, meshPath.string() );

    MyMeanRatioTriangle myMetric;

    DMO::DmoMesh dmoMeshAll = DMO::DmoMesh<true>::create<TriMesh, DMO::Set::All>( mesh );

    std::cout << "Start solver" << std::endl;
    DMO::Solver( mesh, &myMetric, &dmoMeshAll ).solve( 100 );
    std::cout << "Finish solver" << std::endl;

    /* DMO can take two sets of vertices with different metrics. In this case
     * it is not really reasonable to use two sets. It's basically just to show
     * that something like that is possible. DMO toggles between the two sets
     * and metrics performing the number of iterations on both.
     *
     * The following lines of code doe the same as above but the smoothing order
     * will be different because the coloring is computed for each vertex set
     * individually.
     */
    // DMO::DmoMeshCPU dmoMeshInner = DMO::DmoMesh::create<TriMesh, DMO::Set::Inner>( mesh );
    // DMO::DmoMeshCPU dmoMeshBound = DMO::DmoMesh::create<TriMesh, DMO::Set::Boundary>( mesh );
    // DMO::SolverCPU( mesh, &myMetric, &dmoMeshInner, &myMetric, &dmoMeshBound ).solve( 100 );

    OpenMesh::IO::write_mesh( mesh, "examples/out.off" );
}