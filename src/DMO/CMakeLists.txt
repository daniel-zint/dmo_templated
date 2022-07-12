cmake_dependent_option(DMO_USE_CUDA "Generate DMO with CUDA (recommended)" ON "CMAKE_CUDA_COMPILER" OFF)

if(DMO_USE_CUDA)
    # DMO with CUDA

    MESSAGE(STATUS "Build DMO with CUDA")
    #collect everything within this folder
    file(GLOB DMO_SRC_FILES 
        *.cpp 
        *.h
        *.cu
    )

    add_library(DMO STATIC
        ${DMO_SRC_FILES}
    )

    target_include_directories(DMO PRIVATE 
        ### externals ###
        ${OPENMESH_INCLUDE_DIRS}
    )

    target_link_libraries(DMO INTERFACE CUDA::cudart)

    if(OpenMP_CXX_FOUND)
        target_link_libraries(DMO INTERFACE OpenMP::OpenMP_CXX)
    endif()

    target_link_libraries(DMO PRIVATE
        ${OPENMESH_LIBRARIES} 
    )

    target_compile_definitions(DMO PUBLIC -D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING -D_DMO_USE_CUDA)
else()
    # DMO without CUDA as header only library

    MESSAGE(STATUS "Build DMO without CUDA as header only library")

    file(GLOB DMO_SRC_FILES 
        DmoMesh.h
        DmoParams.h
        Metrics.h
        Set.h
        SolverCPU.h
        SolverCPU_impl.h
        types.h
        Vertex.h
    )
    
    add_library(DMO INTERFACE)
    if(MSVC)
        add_custom_target(DMO_headers SOURCES ${DMO_SRC_FILES})
    endif()

    target_link_libraries(DMO INTERFACE ${OPENMESH_LIBRARIES})

    if(OpenMP_CXX_FOUND)
        target_link_libraries(DMO INTERFACE OpenMP::OpenMP_CXX)
    endif()


    target_compile_definitions(DMO INTERFACE -D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING)
endif()

add_library(DMO::DMO ALIAS DMO)
