# ┌─ ONNX Runtime dependency (Phase 1+) ──────────────────────────────────────┐
# │ INTENT: Provide an imported target `onnxruntime::onnxruntime` by downloading │
# │         Microsoft's prebuilt release for the host platform. Gated behind     │
# │         STEMFORGE_WITH_ONNXRUNTIME (OFF in Phase 0), so this only runs once   │
# │         inference work begins.                                               │
# │ NOTE: Untested until Phase 1 flips the option ON. Bump ORT_VERSION and the   │
# │       per-platform archive names together; pin alongside the opset in        │
# │       Models/README.md (see onnx-runtime-cpp skill).                         │
# └────────────────────────────────────────────────────────────────────────────┘
include(FetchContent)

set(ORT_VERSION "1.20.1" CACHE STRING "ONNX Runtime version to fetch")

# Microsoft publishes prebuilt archives per platform/arch. Select the right one.
if(APPLE)
    set(_ort_archive "onnxruntime-osx-arm64-${ORT_VERSION}.tgz")
elseif(WIN32)
    set(_ort_archive "onnxruntime-win-x64-${ORT_VERSION}.zip")
else()
    set(_ort_archive "onnxruntime-linux-x64-${ORT_VERSION}.tgz")
endif()

set(_ort_url "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${_ort_archive}")

FetchContent_Declare(onnxruntime_prebuilt URL "${_ort_url}")
FetchContent_MakeAvailable(onnxruntime_prebuilt)

set(_ort_root "${onnxruntime_prebuilt_SOURCE_DIR}")

# Locate the runtime library inside the extracted archive.
find_library(ORT_LIB
    NAMES onnxruntime
    PATHS "${_ort_root}/lib"
    NO_DEFAULT_PATH)

if(NOT ORT_LIB)
    message(FATAL_ERROR "ONNX Runtime library not found under ${_ort_root}/lib")
endif()

add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${_ort_root}/include")

message(STATUS "ONNX Runtime ${ORT_VERSION} → ${ORT_LIB}")
