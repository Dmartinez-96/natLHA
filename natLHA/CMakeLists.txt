cmake_minimum_required(VERSION 3.10)
project(natLHA)

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(NATLHA_STATIC_LINK "Build fully static natLHA executables" ON)
option(NATLHA_ENABLE_CUDA "Build the CUDA batch backend" OFF)

if(NATLHA_ENABLE_CUDA)
    if(CMAKE_VERSION VERSION_LESS 3.24)
        message(FATAL_ERROR
            "natLHA CUDA builds require CMake 3.24 or newer for the native "
            "architecture default")
    endif()
    if(NATLHA_STATIC_LINK)
        message(FATAL_ERROR
            "NATLHA_ENABLE_CUDA requires -DNATLHA_STATIC_LINK=OFF; fully static CUDA "
            "executables are not a supported configuration")
    endif()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING
            "CUDA architectures for natLHA (defaults to the build machine GPU)")
    endif()
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR
            "NATLHA_ENABLE_CUDA=ON but no CUDA compiler was found; set "
            "CMAKE_CUDA_COMPILER to a CUDA 12.8 or newer nvcc")
    endif()
    enable_language(CUDA)
    # CUDA 12.8 is natLHA's selected CUDA-toolkit baseline.
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
        message(FATAL_ERROR
            "natLHA CUDA builds require CUDA 12.8 or newer (found "
            "${CMAKE_CUDA_COMPILER_VERSION})")
    endif()
    find_package(CUDAToolkit 12.8 REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(NATLHA_RUNTIME_LINK_FLAGS)
if(NATLHA_STATIC_LINK)
    set(Boost_USE_STATIC_LIBS ON)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
    list(APPEND NATLHA_RUNTIME_LINK_FLAGS -static-libgcc -static-libstdc++)
endif()

# CPU-side Boost usage is header-only. CUDA adds the Fiber and Context libraries for
# logical-point scheduling. Filesystem work in terminal_UI.cpp uses std::filesystem, not
# Boost.Filesystem. Boost.System has no library to link against as of Boost 1.90;
# requesting it as a component makes find_package abort. MPFR and GMP are still needed
# at link time, below. To re-derive the header dependencies:
# grep -rhoE "boost/[A-Za-z0-9_/]+\.hpp" src include
if(NATLHA_ENABLE_CUDA)
    find_package(Boost 1.74 REQUIRED COMPONENTS fiber context)
else()
    find_package(Boost 1.74 REQUIRED)
endif()
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_library(MPFR_LIBRARY NAMES mpfr PATH_SUFFIXES lib)
find_library(GMP_LIBRARY NAMES gmp PATH_SUFFIXES lib)

# The physics lives in a library so the interactive UI, non-interactive CLI, and focused
# tests can drive the same calculators. Sources that define main() deliberately remain
# outside NATLHA_CORE_SOURCES because an entry-point object cannot be linked into every
# consumer of the library.
set(NATLHA_CORE_SOURCES
    src/natlha_api.cpp
    src/natlha_batch.cpp
    src/natlha_execution_context.cpp
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

if(NATLHA_ENABLE_CUDA)
    target_sources(natlha_core PRIVATE src/natlha_cuda_backend.cu)
    target_compile_definitions(natlha_core PUBLIC NATLHA_HAS_CUDA=1)
    target_compile_options(natlha_core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=true;--prec-div=true;--prec-sqrt=true>)
    target_link_libraries(natlha_core PUBLIC CUDA::cudart Boost::fiber Boost::context)
endif()

# PUBLIC so that every consumer of the library inherits the include paths and link
# requirements instead of restating them. Adding a new executable then costs one
# add_executable plus one target_link_libraries against natlha_core.
# SLHAea is a single-header Boost-licensed parser. Keep the pinned copy in-tree so a
# clean checkout builds without an undeclared system header or a configure-time fetch.
# Pinned upstream tree:
# https://github.com/fthomas/slhaea/tree/c7e032cd0bac1125f8d7fd253b53cde96e7a904d
target_include_directories(natlha_core PUBLIC
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/slhaea
    ${Boost_INCLUDE_DIRS}
)
target_link_libraries(natlha_core PUBLIC
    Eigen3::Eigen
    ${MPFR_LIBRARY} ${GMP_LIBRARY}
    ${NATLHA_RUNTIME_LINK_FLAGS}
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

add_executable(test-mz-solver src/test_mz_solver.cpp)
target_link_libraries(test-mz-solver PRIVATE natlha_core)
add_test(NAME mz-solver-contract COMMAND test-mz-solver)

add_executable(test-radcorr-fail-closed src/test_radcorr_fail_closed.cpp)
target_link_libraries(test-radcorr-fail-closed PRIVATE natlha_core)
add_test(NAME radcorr-fail-closed COMMAND test-radcorr-fail-closed)

add_executable(test-dbg-adaptive src/test_dbg_adaptive.cpp)
target_link_libraries(test-dbg-adaptive PRIVATE natlha_core)
add_test(NAME dbg-adaptive-contract COMMAND test-dbg-adaptive)

add_executable(test-cli-args src/test_cli_args.cpp)
target_link_libraries(test-cli-args PRIVATE natlha_core)
add_test(NAME cli-argument-contract COMMAND test-cli-args)
add_test(
    NAME cli-output-alias-contract
    COMMAND ${CMAKE_COMMAND}
        -DCLI=$<TARGET_FILE:natlha-cli>
        -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/test_data/joint_qsusy.slha
        -DWORK=${CMAKE_CURRENT_BINARY_DIR}/cli-output-alias-test
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test_cli_output_alias.cmake)

add_executable(test-api-fail-closed src/test_api_fail_closed.cpp)
target_link_libraries(test-api-fail-closed PRIVATE natlha_core)
add_test(NAME api-label-row-fail-closed COMMAND test-api-fail-closed)

add_executable(test-batch-api src/test_batch_api.cpp)
target_link_libraries(test-batch-api PRIVATE natlha_core)
add_test(
    NAME batch-api-contract
    COMMAND test-batch-api ${CMAKE_CURRENT_SOURCE_DIR}/test_data/joint_qsusy.slha)

add_executable(test-execution-context src/test_execution_context.cpp)
target_link_libraries(test-execution-context PRIVATE natlha_core)
add_test(NAME execution-context-contract COMMAND test-execution-context)

if(NATLHA_ENABLE_CUDA)
    add_executable(test-cuda-ode src/test_cuda_ode.cpp)
    target_link_libraries(test-cuda-ode PRIVATE natlha_core)
    add_test(
        NAME cuda-ode-cpu-parity
        COMMAND test-cuda-ode ${CMAKE_CURRENT_SOURCE_DIR}/test_data/joint_qsusy.slha)
endif()

add_executable(test-ewsb-hessian src/test_ewsb_hessian.cpp)
target_link_libraries(test-ewsb-hessian PRIVATE natlha_core)
add_test(NAME ewsb-hessian-squark-contract COMMAND test-ewsb-hessian)

add_executable(test-dsn-exception-policy src/test_dsn_exception_policy.cpp)
target_link_libraries(test-dsn-exception-policy PRIVATE natlha_core)
add_test(NAME dsn-fail-closed-exception-policy COMMAND test-dsn-exception-policy)

add_executable(test-qsusy-root src/test_qsusy_root.cpp)
target_link_libraries(test-qsusy-root PRIVATE natlha_core)
add_test(NAME qsusy-positive-root-contract COMMAND test-qsusy-root)

add_executable(test-joint-qsusy src/test_joint_qsusy.cpp)
target_link_libraries(test-joint-qsusy PRIVATE natlha_core)
add_test(
    NAME joint-qsusy-mu-contract
    COMMAND test-joint-qsusy ${CMAKE_CURRENT_SOURCE_DIR}/test_data/joint_qsusy.slha)

add_test(
    NAME qsusy-cli-plumbing-contract
    COMMAND ${CMAKE_COMMAND}
        -DCLI=$<TARGET_FILE:natlha-cli>
        -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/test_data/joint_qsusy.slha
        -DWORK=${CMAKE_CURRENT_BINARY_DIR}/qsusy-cli-test
        -DCUDA_ENABLED=${NATLHA_ENABLE_CUDA}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test_qsusy_cli.cmake)
