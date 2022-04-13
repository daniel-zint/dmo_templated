#pragma once

#include "DmoVector.h"
#include "Vertex.h"
#include "Set.h"

#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>

namespace DMO {
    template<bool useGPU = true> class DmoMesh {
      public:
        DmoVector<DmoVertex, useGPU> vertices;      // vertices considered for smoothing
        DmoVector<int, useGPU> oneRingVec;          // neighborhood of  vertices
        DmoVector<int, useGPU> coloredVertexIDs;    // vertex coloring

        std::vector<int> colorOffset_;

        DmoMesh() = default;
        DmoMesh( const DmoMesh<useGPU>& m ) {
            vertices.h()         = m.vertices.h();
            oneRingVec.h()       = m.oneRingVec.h();
            coloredVertexIDs.h() = m.coloredVertexIDs.h();
            colorOffset_         = m.colorOffset_;
        }
        DmoMesh( const std::vector<OpenMesh::SmartVertexHandle>& vhs ) {
            copyMeshData( vhs );
            createColoring( vhs );
        }

        template<typename MeshT, int set = Set::Inner> static DmoMesh create( MeshT& mesh ) {
            DmoMesh<useGPU> m;
            m.copyMeshData<MeshT, set>( mesh );
            m.createColoring<MeshT, set>( mesh );
            return m;
        }

        DmoMesh& operator=( const DmoMesh& m ) {
            vertices.h()         = m.vertices.h();
            oneRingVec.h()       = m.oneRingVec.h();
            coloredVertexIDs.h() = m.coloredVertexIDs.h();
            colorOffset_         = m.colorOffset_;
            return *this;
        }

        void copyHostToDevice() {
            vertices.copyHostToDevice();
            oneRingVec.copyHostToDevice();
            coloredVertexIDs.copyHostToDevice();
        }

      private:
        void copyMeshData( const std::vector<OpenMesh::SmartVertexHandle>& vhs ) {
            vertices.h().resize( vhs.size() );
            // compute oneRing size
            int oneRingVecLength = 0;
            for( const auto& vh: vhs ) {
                auto heh            = vh.out().prev().opp().next();
                const auto heh_init = heh;
                do {
                    oneRingVecLength++;
                    heh = heh.next();
                    if( heh.to() == vh ) {
                        heh = heh.opp();
                        if( heh.is_boundary() )
                            break;
                        else
                            heh = heh.next();
                    }
                } while( heh != heh_init );
                oneRingVecLength++;    // additional count s.th. last element is again the first element
            }
            oneRingVec.h().resize( oneRingVecLength );

            int freeVertexCounter = 0;
            int oneRingCounter    = 0;
            for( const auto& vh: vhs ) {
                DmoVertex& v = vertices.h()[freeVertexCounter++];
                v.idx        = vh.idx();
                v.oneRingID  = oneRingCounter;

                auto heh            = vh.out().prev().opp().next();
                const auto heh_init = heh;
                do {
                    oneRingVec.h()[oneRingCounter++] = heh.from().idx();
                    heh                              = heh.next();
                    if( heh.to() == vh ) {
                        heh = heh.opp();
                        if( heh.is_boundary() )
                            break;
                        else
                            heh = heh.next();
                    }
                } while( heh != heh_init );
                if( heh.is_boundary() )
                    oneRingVec.h()[oneRingCounter++] = heh.to().idx();
                else
                    oneRingVec.h()[oneRingCounter++] = heh.from().idx();

                v.oneRingSize = oneRingCounter - v.oneRingID;
                if( v.oneRingSize >= MAX_ONE_RING_SIZE ) {
                    LOG( WARNING ) << "One ring is larger than the maximal allowed one ring size\n"
                                   << "    v.oneRingSize = " << v.oneRingSize << "\n    MAX_ONE_RING_SIZE = " << MAX_ONE_RING_SIZE
                                   << "\nAdjust MAX_ONE_RING_SIZE for DMO to run correctly";
                }
            }
        }

        template<typename MeshT, int set> void copyMeshData( MeshT& mesh ) {
            int nVertices        = 0;
            int oneRingVecLength = 0;

            for( const auto& vh: mesh.vertices() ) {
                if( set == Set::Inner && vh.is_boundary() )
                    continue;
                else if( set == Set::Boundary && !vh.is_boundary() )
                    continue;

                nVertices++;
                auto heh            = vh.out().prev().opp().next();
                const auto heh_init = heh;
                do {
                    oneRingVecLength++;
                    heh = heh.next();
                    if( heh.to() == vh ) {
                        heh = heh.opp();
                        if( heh.is_boundary() )
                            break;
                        else
                            heh = heh.next();
                    }
                } while( heh != heh_init );
                oneRingVecLength++;    // additional count s.th. last element is again the first element
            }

            vertices.h().resize( nVertices );
            oneRingVec.h().resize( oneRingVecLength );

            int freeVertexCounter = 0;
            int oneRingCounter    = 0;
            for( const auto& vh: mesh.vertices() ) {
                if( set == Set::Inner && vh.is_boundary() )
                    continue;
                else if( set == Set::Boundary && !vh.is_boundary() )
                    continue;

                DmoVertex& v = vertices.h()[freeVertexCounter++];
                v.idx        = vh.idx();
                v.oneRingID  = oneRingCounter;

                auto heh            = vh.out().prev().opp().next();
                const auto heh_init = heh;
                do {
                    oneRingVec.h()[oneRingCounter++] = heh.from().idx();
                    heh                              = heh.next();
                    if( heh.to() == vh ) {
                        heh = heh.opp();
                        if( heh.is_boundary() )
                            break;
                        else
                            heh = heh.next();
                    }
                } while( heh != heh_init );
                if( heh.is_boundary() )
                    oneRingVec.h()[oneRingCounter++] = heh.to().idx();
                else
                    oneRingVec.h()[oneRingCounter++] = heh.from().idx();

                v.oneRingSize = oneRingCounter - v.oneRingID;
                if( v.oneRingSize >= MAX_ONE_RING_SIZE ) {
                    LOG( WARNING ) << "One ring is larger than the maximal allowed one ring size\n"
                                   << "    v.oneRingSize = " << v.oneRingSize << "\n    MAX_ONE_RING_SIZE = " << MAX_ONE_RING_SIZE
                                   << "\nAdjust MAX_ONE_RING_SIZE for DMO to run correctly";
                }
            }
        }

        void createColoring( const std::vector<OpenMesh::SmartVertexHandle>& vhs ) {
            const auto& mesh = vhs[0].mesh();

            // create coloring scheme
            std::vector<int> colorScheme( mesh->n_vertices(), -2 );

            for( const auto& vh: vhs ) {
                colorScheme[vh.idx()] = -1;
            }

            for( const auto& vh: vhs ) {
                unsigned long colorBits = 0;
                auto heh                = vh.out().prev().opp().next();
                const auto heh_init     = heh;
                do {
                    int c = colorScheme[heh.from().idx()];
                    if( c >= 0 )
                        colorBits |= 1 << c;
                    heh = heh.next();
                    if( heh.to() == vh ) {
                        heh = heh.opp();
                        if( heh.is_boundary() )
                            break;
                        else
                            heh = heh.next();
                    }
                } while( heh != heh_init );

                int color = 0;
                while( ( colorBits & ( 1 << color ) ) ) {
                    ++color;
                }
                colorScheme[vh.idx()] = color;
            }

            int n_colors = *( std::max_element( colorScheme.begin(), colorScheme.end() ) ) + 1;

            if( n_colors == -1 )
                return;

            std::vector<int> n_color_vecs( n_colors, 0 );
            for( int i = 0; i < colorScheme.size(); ++i ) {
                if( colorScheme[i] > -1 )
                    ++n_color_vecs[colorScheme[i]];
            }

            coloredVertexIDs.h().resize( vhs.size() );

            colorOffset_.resize( n_colors + 1, 0 );
            for( int i = 1; i < n_colors; ++i ) {
                colorOffset_[i] = colorOffset_[i - 1] + n_color_vecs[i - 1];
            }
            colorOffset_[n_colors] = static_cast<int>( vhs.size() );    // mark the end of the colored-vertices vector

            // add vertex ids
            std::vector<int> colorCounter( n_colors, 0 );
            int interior_counter = 0;
            for( int i = 0; i < colorScheme.size(); ++i ) {
                if( colorScheme[i] < 0 ) {
                    continue;
                }
                coloredVertexIDs.h()[colorOffset_[colorScheme[i]] + colorCounter[colorScheme[i]]++] = interior_counter++;
            }
        }

        template<typename MeshT, int set> void createColoring( MeshT& mesh ) {
            // create coloring scheme
            std::vector<int> colorScheme( mesh.n_vertices(), -2 );

            for( const auto& vh: mesh.vertices() ) {
                if( set == Set::Inner && vh.is_boundary() )
                    continue;
                else if( set == Set::Boundary && !vh.is_boundary() )
                    continue;

                colorScheme[vh.idx()] = -1;
            }

            for( const auto& vh: mesh.vertices() ) {
                if( set == Set::Inner && vh.is_boundary() )
                    continue;
                else if( set == Set::Boundary && !vh.is_boundary() )
                    continue;

                unsigned long colorBits = 0;
                auto heh                = vh.out().prev().opp().next();
                const auto heh_init     = heh;
                do {
                    int c = colorScheme[heh.from().idx()];
                    if( c >= 0 )
                        colorBits |= 1 << c;
                    heh = heh.next();
                    if( heh.to() == vh ) {
                        heh = heh.opp();
                        if( heh.is_boundary() )
                            break;
                        else
                            heh = heh.next();
                    }
                } while( heh != heh_init );

                int color = 0;
                while( ( colorBits & ( 1 << color ) ) ) {
                    ++color;
                }
                colorScheme[vh.idx()] = color;
            }

            int n_colors = *( std::max_element( colorScheme.begin(), colorScheme.end() ) ) + 1;

            if( n_colors == -1 )
                return;

            std::vector<int> n_color_vecs( n_colors, 0 );
            for( int i = 0; i < colorScheme.size(); ++i ) {
                if( colorScheme[i] > -1 )
                    ++n_color_vecs[colorScheme[i]];
            }

            coloredVertexIDs.h().resize( vertices.h().size() );

            colorOffset_.resize( n_colors + 1, 0 );
            for( int i = 1; i < n_colors; ++i ) {
                colorOffset_[i] = colorOffset_[i - 1] + n_color_vecs[i - 1];
            }
            colorOffset_[n_colors] = static_cast<int>( vertices.h().size() );    // mark the end of the colored-vertices vector

            // add vertex ids
            std::vector<int> colorCounter( n_colors, 0 );
            int interior_counter = 0;
            for( int i = 0; i < colorScheme.size(); ++i ) {
                if( colorScheme[i] < 0 ) {
                    continue;
                }
                coloredVertexIDs.h()[colorOffset_[colorScheme[i]] + colorCounter[colorScheme[i]]++] = interior_counter++;
            }
        }
    };
}    // namespace DMO