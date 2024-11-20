add_subdirectory(DMO)

add_subdirectory(ex01_Basic_cpu)
add_subdirectory(ex02_UserMetric_cpu)

if(DMO_USE_CUDA)
    add_subdirectory(ex01_Basic)
    add_subdirectory(ex02_UserMetric)
endif()

add_subdirectory(ex03_easy_switch)