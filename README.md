# Discrete Mesh Optimization

Discrete Mesh Optimization (DMO) implemented in C++ and CUDA for 2D meshes. The method works on triangle and quad meshes. Some metrics are already given but the power of DMO is that own metrics can be defined also.

## Requirements
- C++17
- OpenMesh 8.1 or higher: https://www.openmesh.org

Cuda is recommended for good performance. If no Cuda Compiler is found, DMO will be set up as a header only library.

## Installation

### Windows
- Download and install CMake (at least Version 3.18)
- Start CMake Gui and choose *dmo_templated* as Source Directory.
- Choose a Build Directory, e.g. create a new Folder "build" in *dmo_templated*.
- Click "add Entry" and enter "CMAKE_INSTALL_PREFIX". Select the path to the installation of OpenMesh if it has not been installed to the default location.
- Click "Configure", once finished click "Generate".
- Click "Open Project"
- Select a Startup Project, e.g. **ex01_Basic** or **ex01_Basic_cpu**, and build it.

### Linux (not tested)
- Create a build folder in the root directory of the project.
- Execute "cmake .. -DCMAKE_INSTALL_PREFIX=x" inside the build folder where x is the location of the installation of OpenMesh.
- call make.

## Getting started
Have a look at the provided examples. Start with ex01_Basic to see how DMO is used with the predefined metrics. For using your own metrics take a look at ex02_UserMetric.

## Usage in other CMake projects
- Copy the **DMO** folder to your own CMake project.
- Add *add_subdirectory(DMO)* to your CMake project.
- Add package **DMO::DMO** to your *target_link_libraries()*