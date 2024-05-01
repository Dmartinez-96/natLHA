cmake_minimum_required(VERSION 3.10)  
project(DSN4SLHA)                       

set(CMAKE_CXX_STANDARD 17)            
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen, Boost, and GSL.
find_package(Boost 1.74 REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(GSL 2.5 REQUIRED)
find_library(MPFR_LIBRARY NAMES mpfr)
find_library(GMP_LIBRARY NAMES gmp)

# Source files
set(SOURCE_FILES
    src/terminal_UI.cpp
    src/radcorr_calc.cpp
    src/MSSM_RGE_solver.cpp
    src/MSSM_RGE_solver_with_stopfinder.cpp
    src/mZ_numsolver.cpp
    src/DSN_calc.cpp
    src/tree_mass_calc.cpp
    src/EWSB_loop.cpp
)

# Executable
add_executable(DSN4SLHA ${SOURCE_FILES})

# Include directories
target_include_directories(DSN4SLHA PRIVATE include ${Boost_INCLUDE_DIRS})

# Link libraries
target_link_libraries(DSN4SLHA Eigen3::Eigen GSL::gsl GSL::gslcblas ${MPFR_LIBRARY} ${GMP_LIBRARY})