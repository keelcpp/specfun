cmake_minimum_required(VERSION 3.20)
project(keel_specfun VERSION 0.1.0 LANGUAGES CXX)

option(KEEL_SPECFUN_NO_EXCEPTIONS_RTTI "Compile keel_specfun without C++ exceptions and RTTI" OFF)
option(KEEL_SPECFUN_BUILD_TESTS "Build keel_specfun unit tests and header purity checks" ${PROJECT_IS_TOP_LEVEL})

if(NOT TARGET keel::keel)
    find_package(keel QUIET CONFIG)
endif()
if(NOT TARGET keel::keel)
    include(FetchContent)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../keel/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL)
        set(FETCHCONTENT_SOURCE_DIR_KEEL "${CMAKE_CURRENT_SOURCE_DIR}/../keel" CACHE PATH "Local keel checkout")
    endif()
    if(KEEL_SPECFUN_NO_EXCEPTIONS_RTTI)
        set(KEEL_NO_EXCEPTIONS_RTTI ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(
        keel
        GIT_REPOSITORY https://github.com/keelcpp/keel.git
        GIT_TAG        869a5c07a5d77341e5f6acbac5f8e024b274ea08
    )
    FetchContent_MakeAvailable(keel)
endif()

add_library(keel_specfun STATIC
    src/specfun.cpp
    src/bessel.cpp
    src/combinatorics.cpp
)
add_library(keel::specfun ALIAS keel_specfun)

target_link_libraries(keel_specfun PUBLIC keel::keel)

target_compile_features(keel_specfun PUBLIC cxx_std_20)
set_target_properties(keel_specfun PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(keel_specfun PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(MSVC)
    target_compile_options(keel_specfun PRIVATE /W4 /utf-8)
    if(KEEL_SPECFUN_NO_EXCEPTIONS_RTTI)
        string(REGEX REPLACE "/EH[a-z]+" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        string(REGEX REPLACE "/GR" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        target_compile_options(keel_specfun PUBLIC /EHs-c- /GR-)
        target_compile_definitions(keel_specfun PUBLIC _HAS_EXCEPTIONS=0)
    endif()
else()
    target_compile_options(keel_specfun PRIVATE -Wall -Wextra -Wpedantic)
    if(KEEL_SPECFUN_NO_EXCEPTIONS_RTTI)
        target_compile_options(keel_specfun PUBLIC -fno-exceptions -fno-rtti)
    endif()
endif()

if(COMMAND keel_enforce_guidelines)
    keel_enforce_guidelines(keel_specfun
        SCOPE ALL
        BAN_RAW_INTS ALL
        BAN_RAW_FLOATS ALL
        HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include"
        STANDALONE_HEADERS ${KEEL_SPECFUN_BUILD_TESTS}
        TEST_NAME test_guidelines_headers
        WARNINGS OFF
    )
endif()

include(CMakePackageConfigHelpers)
install(TARGETS keel_specfun EXPORT keel_specfunTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/keel DESTINATION include)
install(EXPORT keel_specfunTargets
    FILE keel_specfunTargets.cmake
    NAMESPACE keel::
    DESTINATION lib/cmake/keel_specfun
)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/keel_specfunConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/keel_specfunConfig.cmake"
    INSTALL_DESTINATION lib/cmake/keel_specfun
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/keel_specfunConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/keel_specfunConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/keel_specfunConfigVersion.cmake"
    DESTINATION lib/cmake/keel_specfun
)

if(KEEL_SPECFUN_BUILD_TESTS)
    enable_testing()

    add_executable(test_specfun tests/test_specfun.cpp)
    target_link_libraries(test_specfun PRIVATE keel::specfun)
    if(MSVC)
        target_compile_options(test_specfun PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_specfun PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME test_specfun COMMAND test_specfun)

    get_target_property(KEEL_CORE_INC_DIRS keel::keel INTERFACE_INCLUDE_DIRECTORIES)
    set(KEEL_CORE_INC_CLEAN "")
    foreach(IDIR IN LISTS KEEL_CORE_INC_DIRS)
        string(REGEX REPLACE "\\$<BUILD_INTERFACE:([^>]+)>" "\\1" IDIR_STRIPPED "${IDIR}")
        if(NOT IDIR_STRIPPED MATCHES "\\$<INSTALL_INTERFACE:" AND EXISTS "${IDIR_STRIPPED}")
            list(APPEND KEEL_CORE_INC_CLEAN "${IDIR_STRIPPED}")
        endif()
    endforeach()
    string(REPLACE ";" "|" KEEL_DEP_INC_JOINED "${KEEL_CORE_INC_CLEAN}")
    string(REPLACE ";" "|" KEEL_STD_INC_JOINED "${CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES}")

    add_test(
        NAME test_header_purity
        COMMAND ${CMAKE_COMMAND}
            "-DKEEL_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/include"
            "-DKEEL_DEP_INC_DIRS=${KEEL_DEP_INC_JOINED}"
            "-DCOMPILER_PATH=${CMAKE_CXX_COMPILER}"
            "-DCOMPILER_ID=${CMAKE_CXX_COMPILER_ID}"
            "-DSTD_INC_DIRS=${KEEL_STD_INC_JOINED}"
            "-DCMAKE_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckHeaderPurity.cmake"
    )
endif()
