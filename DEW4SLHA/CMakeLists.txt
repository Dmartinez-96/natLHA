cmake_minimum_required(VERSION 3.10)  
project(DEW4SLHA)                       

set(CMAKE_CXX_STANDARD 17)            
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen, Boost, and GSL.
find_package(Boost 1.74 REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(GSL 2.7.1 REQUIRED)

# Source files
set(SOURCE_FILES
    src/terminal_UI.cpp
    src/radcorr_calc.cpp
    src/MSSM_RGE_solver.cpp
    src/MSSM_RGE_solver_with_stopfinder.cpp
    src/DEW_calc.cpp
    src/mZ_numsolver.cpp
    src/tree_mass_calc.cpp
    src/EWSB_loop.cpp
)

# Executable
add_executable(DEW4SLHA ${SOURCE_FILES})

# Include directories
target_include_directories(DEW4SLHA PRIVATE include ${Boost_INCLUDE_DIRS})

# Link libraries
target_link_libraries(DEW4SLHA Eigen3::Eigen GSL::gsl GSL::gslcblas)