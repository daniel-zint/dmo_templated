set(BASE_DIRECTORY "${CMAKE_SOURCE_DIR}")
configure_file(baseDirectory.h.in baseDirectory.h)

set(EXEC_NAME ex01_Basic_cpu)

file(GLOB EXEC_FILES *.cpp *.h)

#add project executable
add_executable(${EXEC_NAME}
	${EXEC_FILES}
)

target_include_directories(${EXEC_NAME} PRIVATE 
	${SRC}
    ${CMAKE_CURRENT_BINARY_DIR}
    ### externals ###
    ${OPENMESH_INCLUDE_DIRS}
)

target_link_libraries(${EXEC_NAME} PRIVATE 
	DMO::DMO
    ${LIBS} 
)
target_compile_definitions(${EXEC_NAME} PUBLIC -D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING)

if(MSVC)
    set_target_properties(${EXEC_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
endif()
