cmake_minimum_required(VERSION 3.10)
project(natLHA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prefer the static libraries of all dependencies
set(Boost_USE_STATIC_LIBS ON)  
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
set(CMAKE_EXE_LINKER_FLAGS "-static")

find_package(Boost 1.74 REQUIRED COMPONENTS system filesystem)  # Specify the required Boost components
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(GSL 2.5 REQUIRED)
find_library(MPFR_LIBRARY NAMES mpfr PATH_SUFFIXES lib)
find_library(GMP_LIBRARY NAMES gmp PATH_SUFFIXES lib)

# Source files
set(SOURCE_FILES
    src/terminal_UI.cpp
    src/radcorr_calc.cpp
    src/MSSM_RGE_solver.cpp
    src/MSSM_RGE_solver_with_stopfinder.cpp
    src/DEW_calc.cpp
    src/DHS_calc.cpp
    src/DBG_calc.cpp
    src/mZ_numsolver.cpp
    src/DSN_calc.cpp
    src/tree_mass_calc.cpp
    src/EWSB_loop.cpp
    src/other_derivs.cpp
)

# Executable
add_executable(natLHA ${SOURCE_FILES})

# Include directories
target_include_directories(natLHA PRIVATE include ${Boost_INCLUDE_DIRS})

# Link all libraries statically
target_link_libraries(natLHA
    Eigen3::Eigen
    GSL::gsl GSL::gslcblas
    ${MPFR_LIBRARY} ${GMP_LIBRARY}
    ${Boost_LIBRARIES}
    -static-libgcc -static-libstdc++
)

# Remove RPATH to avoid system paths in the executable
set_target_properties(natLHA PROPERTIES INSTALL_RPATH_USE_LINK_PATH FALSE)
