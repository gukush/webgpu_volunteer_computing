cmake_minimum_required(VERSION 3.20)
project(MultiFrameworkClient LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable vcpkg toolchain integration
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "")
endif()

# Find required packages via vcpkg
find_package(Boost REQUIRED COMPONENTS system url)
find_package(OpenSSL REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# Optional: CUDA support
option(ENABLE_CUDA "Enable CUDA support" ON)
if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA found - enabling CUDA executor")
        set(HAVE_CUDA ON)
    else()
        message(WARNING "CUDA not found - disabling CUDA executor")
        set(HAVE_CUDA OFF)
    endif()
endif()

if(NOT TARGET CUDA::cudart)
  # Try to locate libcudart
  find_library(CUDART_LIBRARY NAMES cudart libcudart PATHS
    /usr/local/cuda/lib64
    /usr/local/cuda-*/targets/x86_64-linux/lib
    /usr/local/cuda-*/lib64
    NO_DEFAULT_PATH
  )

  # also pick up common cuda include path
  find_path(CUDART_INCLUDE_DIR cuda_runtime.h PATHS
    /usr/local/cuda/include
    /usr/local/cuda-*/targets/x86_64-linux/include
    NO_DEFAULT_PATH
  )

  if(CUDART_LIBRARY)
    message(STATUS "Creating imported target CUDA::cudart -> ${CUDART_LIBRARY}")
    add_library(CUDA::cudart UNKNOWN IMPORTED)
    set_target_properties(CUDA::cudart PROPERTIES
      IMPORTED_LOCATION "${CUDART_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${CUDART_INCLUDE_DIR}"
    )
  else()
    message(WARNING "Could not find libcudart; CUDA runtime functions may link-fail")
  endif()
endif()

if(NOT TARGET CUDA::nvrtc)
  message(STATUS "CUDA::nvrtc target not found — attempting manual nvrtc detection")

  # likely CUDA install prefixes on this machine
  set(_cuda_hint_paths
    $ENV{CUDA_HOME}
    /usr/local/cuda-13
    /usr/local/cuda
    /usr
    /usr/local
  )

  # find nvrtc header (nvrtc.h)
  find_path(NVRTC_INCLUDE_DIR nvrtc.h
    HINTS ${_cuda_hint_paths}
    PATH_SUFFIXES include include/cuda include/x86_64-linux-gnu/include
  )

  # find nvrtc library (try sonames and generic name)
  find_library(NVRTC_LIBRARY NAMES nvrtc nvrtc.so nvrtc.so.13 nvrtc.so.12
    HINTS ${_cuda_hint_paths}
    PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib lib/x86_64-linux-gnu
  )

  if(NVRTC_LIBRARY AND NVRTC_INCLUDE_DIR)
    message(STATUS "Found nvrtc: ${NVRTC_LIBRARY}; include: ${NVRTC_INCLUDE_DIR}")
    # create an IMPORTED target so other code can link to CUDA::nvrtc as usual
    add_library(CUDA::nvrtc SHARED IMPORTED)
    set_target_properties(CUDA::nvrtc PROPERTIES
      IMPORTED_LOCATION "${NVRTC_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${NVRTC_INCLUDE_DIR}"
    )
  else()
    message(WARNING "nvrtc library or header not found; CUDA NVRTC functionality will be unavailable.")
  endif()
endif()

# Optional: OpenCL support
option(ENABLE_OPENCL "Enable OpenCL support" ON)
if(ENABLE_OPENCL)
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        message(STATUS "OpenCL found - enabling OpenCL executor")
        set(HAVE_OPENCL ON)
    else()
        message(WARNING "OpenCL not found - disabling OpenCL executor")
        set(HAVE_OPENCL OFF)
    endif()
endif()

if(NOT TARGET OpenCL::OpenCL)
  if(DEFINED OpenCL_LIBRARY AND OpenCL_LIBRARY)
    message(STATUS "Creating imported target OpenCL::OpenCL -> ${OpenCL_LIBRARY}")
    add_library(OpenCL::OpenCL UNKNOWN IMPORTED)
    set_target_properties(OpenCL::OpenCL PROPERTIES
      IMPORTED_LOCATION "${OpenCL_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${OpenCL_INCLUDE_DIR}"
    )
  else()
    message(WARNING "OpenCL library not provided; OpenCL executor will be disabled")
  endif()
endif()


if(OpenCL_FOUND)
  message(STATUS "OpenCL found: ${OpenCL_LIBRARY}; include: ${OpenCL_INCLUDE_DIR}")

  if(TARGET OpenCL::OpenCL)
    message(STATUS "Using imported target OpenCL::OpenCL")
  endif()

  # fallback: add include directory globally so headers like <CL/cl.h> resolve
  if(DEFINED OpenCL_INCLUDE_DIR AND OpenCL_INCLUDE_DIR)
    include_directories(BEFORE "${OpenCL_INCLUDE_DIR}")
  endif()
else()
  message(WARNING "OpenCL not found (headers or lib); OpenCL targets will be disabled")
endif()

# Optional: Vulkan support
option(ENABLE_VULKAN "Enable Vulkan support" ON)
if(ENABLE_VULKAN)
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        message(STATUS "Vulkan found - enabling Vulkan executor")
        set(HAVE_VULKAN ON)
    else()
        message(WARNING "Vulkan not found - disabling Vulkan executor")
        set(HAVE_VULKAN OFF)
    endif()
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Common source files
set(COMMON_SOURCES
    common/framework_client.cpp
    common/websocket_client.cpp
    common/base64.cpp
)

# Function to create executables for each framework
function(create_framework_executable framework_name sources link_libs compile_defs)
    add_executable(${framework_name}_client
        main.cpp
        ${COMMON_SOURCES}
        ${sources}
    )

    target_link_libraries(${framework_name}_client
        PRIVATE
        Boost::system
        Boost::url
        OpenSSL::SSL
        OpenSSL::Crypto
        nlohmann_json::nlohmann_json
        ${link_libs}
        ${CMAKE_THREAD_LIBS_INIT}
    )

    target_compile_definitions(${framework_name}_client PRIVATE ${compile_defs})

    # Set output directory
    set_target_properties(${framework_name}_client PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    message(STATUS "Created target: ${framework_name}_client")
endfunction()

# Find threads (required for std::thread)
find_package(Threads REQUIRED)


if(HAVE_CUDA)
  # common CUDA include locations (adjust if needed)
  foreach(_p
      $ENV{CUDA_HOME}
      /usr/local/cuda-13/targets/x86_64-linux/include
      /usr/local/cuda-13/include
      /usr/local/cuda/include
  )
    if(_p AND EXISTS "${_p}")
      message(STATUS "Adding CUDA include dir: ${_p}")
      include_directories(BEFORE "${_p}")
      break()
    endif()
  endforeach()
endif()

if(HAVE_CUDA)
    list(APPEND UNIVERSAL_LIBS CUDA::cuda_driver CUDA::nvrtc CUDA::cudart)
endif()

# CUDA Client
if(HAVE_CUDA)
    create_framework_executable("cuda"
        "cuda/cuda_executor.cpp"
        "CUDA::cuda_driver;CUDA::nvrtc;CUDA::cudart"
        "HAVE_CUDA=1"
    )
endif()

# OpenCL Client
if(HAVE_OPENCL)
    create_framework_executable("opencl"
        "opencl/opencl_executor.cpp"
        "OpenCL::OpenCL"
        "HAVE_OPENCL=1"
    )
endif()

# Vulkan Client
if(HAVE_VULKAN)
    create_framework_executable("vulkan"
        "vulkan/vulkan_executor.cpp"
        "Vulkan::Vulkan"
        "HAVE_VULKAN=1"
    )
endif()

# Universal client that can be built with all available frameworks
set(UNIVERSAL_SOURCES ${COMMON_SOURCES})
set(UNIVERSAL_LIBS Boost::system Boost::url OpenSSL::SSL OpenSSL::Crypto nlohmann_json::nlohmann_json ${CMAKE_THREAD_LIBS_INIT})
set(UNIVERSAL_DEFS "")

if(HAVE_CUDA)
    list(APPEND UNIVERSAL_SOURCES cuda/cuda_executor.cpp)
    list(APPEND UNIVERSAL_LIBS CUDA::cuda_driver CUDA::nvrtc)
    list(APPEND UNIVERSAL_DEFS HAVE_CUDA=1)
endif()

if(HAVE_OPENCL)
    list(APPEND UNIVERSAL_SOURCES opencl/opencl_executor.cpp)
    list(APPEND UNIVERSAL_LIBS OpenCL::OpenCL)
    list(APPEND UNIVERSAL_DEFS HAVE_OPENCL=1)
endif()

if(HAVE_VULKAN)
    list(APPEND UNIVERSAL_SOURCES vulkan/vulkan_executor.cpp)
    list(APPEND UNIVERSAL_LIBS Vulkan::Vulkan)
    list(APPEND UNIVERSAL_DEFS HAVE_VULKAN=1)
endif()

# Only create universal client if at least one framework is available
if(HAVE_CUDA OR HAVE_OPENCL OR HAVE_VULKAN)
    add_executable(universal_client
        main.cpp
        ${UNIVERSAL_SOURCES}
    )

    target_link_libraries(universal_client
        PRIVATE
        ${UNIVERSAL_LIBS}
    )

    target_compile_definitions(universal_client PRIVATE ${UNIVERSAL_DEFS})

    set_target_properties(universal_client PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    message(STATUS "Created target: universal_client")
else()
    message(WARNING "No compute frameworks found - no executables will be built")
endif()

# Print build summary
message(STATUS "=== Build Summary ===")
message(STATUS "CUDA Support: ${HAVE_CUDA}")
message(STATUS "OpenCL Support: ${HAVE_OPENCL}")
message(STATUS "Vulkan Support: ${HAVE_VULKAN}")
message(STATUS "====================")

# Installation rules
set(INSTALL_TARGETS universal_client)

if(HAVE_CUDA)
    list(APPEND INSTALL_TARGETS cuda_client)
endif()

if(HAVE_OPENCL)
    list(APPEND INSTALL_TARGETS opencl_client)
endif()

if(HAVE_VULKAN)
    list(APPEND INSTALL_TARGETS vulkan_client)
endif()

if(HAVE_CUDA OR HAVE_OPENCL OR HAVE_VULKAN)
    install(TARGETS ${INSTALL_TARGETS}
        DESTINATION bin
    )
endif()