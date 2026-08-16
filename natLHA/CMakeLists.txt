cmake_minimum_required(VERSION 3.10)
project(natLHA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prefer the static libraries of all dependencies
set(Boost_USE_STATIC_LIBS ON)  
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
set(CMAKE_EXE_LINKER_FLAGS "-static")

# No COMPONENTS: every Boost library natLHA includes is header-only (math/tools/roots,
# math/special_functions/next, multiprecision/mpfr, multiprecision/eigen, numeric/odeint).
# Filesystem work uses std::filesystem, not Boost.Filesystem -- see terminal_UI.cpp:35.
# Boost.System has no library to link against as of Boost 1.90; requesting it as a
# component makes find_package abort. MPFR and GMP are still needed at link time, below.
# To re-derive: grep -rhoE "boost/[A-Za-z0-9_/]+\.hpp" src include
find_package(Boost 1.74 REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(GSL 2.5 REQUIRED)
find_library(MPFR_LIBRARY NAMES mpfr PATH_SUFFIXES lib)
find_library(GMP_LIBRARY NAMES gmp PATH_SUFFIXES lib)

# The physics lives in a library, not in an executable, so that more than one program can
# drive the same calculators: the interactive terminal UI, and a non-interactive CLI for
# batch label generation. Exactly one source in src/ defines main() -- main_interactive.cpp,
# confirmed by grepping all of src/*.cpp for a main definition -- and it is deliberately
# NOT in the list below, because an object defining main() cannot be linked into a second
# executable. That constraint is what forces this split.
set(NATLHA_CORE_SOURCES
    src/natlha_api.cpp
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

add_library(natlha_core STATIC ${NATLHA_CORE_SOURCES})

# PUBLIC so that every consumer of the library inherits the include paths and link
# requirements instead of restating them. Adding a new executable then costs one
# add_executable plus one target_link_libraries against natlha_core.
target_include_directories(natlha_core PUBLIC include ${Boost_INCLUDE_DIRS})
target_link_libraries(natlha_core PUBLIC
    Eigen3::Eigen
    GSL::gsl GSL::gslcblas
    ${MPFR_LIBRARY} ${GMP_LIBRARY}
    -static-libgcc -static-libstdc++
)

# The interactive program: entry point only, all behaviour from the library.
add_executable(natLHA src/main_interactive.cpp)
target_link_libraries(natLHA PRIVATE natlha_core)

# The non-interactive front end. Same library, no prompts, parseable output.
add_executable(natlha-cli src/main_cli.cpp)
target_link_libraries(natlha-cli PRIVATE natlha_core)
set_target_properties(natlha-cli PROPERTIES INSTALL_RPATH_USE_LINK_PATH FALSE)

# Remove RPATH to avoid system paths in the executable
set_target_properties(natLHA PROPERTIES INSTALL_RPATH_USE_LINK_PATH FALSE)

enable_testing()
add_executable(test-delta-sn src/test_delta_sn.cpp)
target_link_libraries(test-delta-sn PRIVATE natlha_core)
add_test(NAME delta-sn-eq-5.21 COMMAND test-delta-sn)
