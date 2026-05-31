# ┌─ StemForge — root build config ───────────────────────────────────────────┐
# │ INTENT: Build the StemForge plugin (VST3 + Standalone) from one source tree │
# │         using JUCE 8 pulled via FetchContent. ONNX Runtime is wired but      │
# │         OFF by default (Phase 0 has no inference yet — see cmake/).          │
# └──────────────────────────────────────────────────────────────────────────────┘
cmake_minimum_required(VERSION 3.22)

project(StemForge VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Apple Silicon by default; override with -DCMAKE_OSX_ARCHITECTURES=... for universal.
if(APPLE AND NOT CMAKE_OSX_ARCHITECTURES)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "" FORCE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "" FORCE)
endif()

# ── Options ───────────────────────────────────────────────────────────────────
# OFF for Phase 0; flip ON in Phase 1 to fetch + link ONNX Runtime.
option(STEMFORGE_WITH_ONNXRUNTIME "Fetch and link ONNX Runtime (Phase 1+)" OFF)

# ── JUCE via FetchContent (never committed — see .gitignore) ───────────────────
include(FetchContent)
FetchContent_Declare(juce
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG        8.0.13
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(juce)

# ── The plugin target: one source tree → VST3 + Standalone ─────────────────────
# AU is added in Phase 4. Codes must be unique 4-char identifiers.
juce_add_plugin(StemForge
    PRODUCT_NAME              "StemForge"
    COMPANY_NAME              "StemForge"
    BUNDLE_ID                 "com.stemforge.stemforge"
    PLUGIN_MANUFACTURER_CODE  Stmf
    PLUGIN_CODE               Stfg
    FORMATS                   VST3 Standalone
    IS_SYNTH                  FALSE
    NEEDS_MIDI_INPUT          FALSE
    NEEDS_MIDI_OUTPUT         FALSE
    IS_MIDI_EFFECT            FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD   FALSE)

target_sources(StemForge PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
    Source/UI/FileDropZone.cpp
    Source/IO/AudioFileIO.cpp
    Source/Separation/AudioChunker.cpp
    Source/Separation/StemWriter.cpp)

target_include_directories(StemForge PRIVATE Source)

# JUCE feature toggles — trim what we don't use (no browser, no curl).
target_compile_definitions(StemForge PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0)

target_link_libraries(StemForge
    PRIVATE
        juce::juce_audio_utils      # pulls audio_processors, audio_formats, gui_basics, …
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags)

# ── ONNX Runtime (Phase 1+, gated OFF) ─────────────────────────────────────────
# The inference engine only compiles when ORT is available (it includes Ort headers),
# so its source and the headless separation driver live inside this block.
if(STEMFORGE_WITH_ONNXRUNTIME)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/onnxruntime.cmake)
    target_sources(StemForge PRIVATE
        Source/Separation/ONNXInferenceEngine.cpp
        Source/Separation/SeparationWorker.cpp)
    target_link_libraries(StemForge PRIVATE onnxruntime::onnxruntime)
    target_compile_definitions(StemForge PRIVATE
        STEMFORGE_HAS_ONNX=1
        # Phase-2 dev default model (proper bundling is Phase 4). Overridable at runtime via
        # the STEMFORGE_MODEL_PATH env var — see StemForgeProcessor::getDefaultModelFile().
        STEMFORGE_DEFAULT_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/Models/weights/htdemucs.onnx")

    # Headless Phase 1 harness: model + audio file in → stem WAVs out. The end-to-end
    # integration of chunker → engine → overlap-add → writer, runnable from a shell.
    juce_add_console_app(StemForgeSeparate PRODUCT_NAME "StemForgeSeparate")
    target_sources(StemForgeSeparate PRIVATE
        Tests/SeparateCli.cpp
        Source/Separation/AudioChunker.cpp
        Source/Separation/StemWriter.cpp
        Source/Separation/ONNXInferenceEngine.cpp
        Source/IO/AudioFileIO.cpp)
    target_include_directories(StemForgeSeparate PRIVATE Source)
    target_compile_definitions(StemForgeSeparate PRIVATE
        JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 STEMFORGE_HAS_ONNX=1)
    target_link_libraries(StemForgeSeparate
        PRIVATE
            juce::juce_audio_formats
            onnxruntime::onnxruntime
        PUBLIC
            juce::juce_recommended_config_flags
            juce::juce_recommended_warning_flags)

    # Phase-2 verification harness: drives the real SeparationWorker (thread + progress +
    # AsyncUpdater completion) under a live MessageManager — the path the Analyze button hits.
    juce_add_console_app(StemForgeWorkerDriver PRODUCT_NAME "StemForgeWorkerDriver")
    target_sources(StemForgeWorkerDriver PRIVATE
        Tests/WorkerDriver.cpp
        Source/Separation/SeparationWorker.cpp
        Source/Separation/AudioChunker.cpp
        Source/Separation/StemWriter.cpp
        Source/Separation/ONNXInferenceEngine.cpp
        Source/IO/AudioFileIO.cpp)
    target_include_directories(StemForgeWorkerDriver PRIVATE Source)
    target_compile_definitions(StemForgeWorkerDriver PRIVATE
        JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 STEMFORGE_HAS_ONNX=1
        JUCE_MODAL_LOOPS_PERMITTED=1)   # enables runDispatchLoopUntil for the pump loop
    target_link_libraries(StemForgeWorkerDriver
        PRIVATE
            juce::juce_audio_formats
            juce::juce_events
            onnxruntime::onnxruntime
        PUBLIC
            juce::juce_recommended_config_flags
            juce::juce_recommended_warning_flags)
endif()

# ── Tests ──────────────────────────────────────────────────────────────────────
# Headless console app that asserts the Phase 0 lossless-roundtrip contract via ctest.
enable_testing()

juce_add_console_app(StemForgeRoundtripTest PRODUCT_NAME "StemForgeRoundtripTest")
target_sources(StemForgeRoundtripTest PRIVATE
    Tests/RoundtripTest.cpp
    Source/IO/AudioFileIO.cpp)
target_include_directories(StemForgeRoundtripTest PRIVATE Source)
target_compile_definitions(StemForgeRoundtripTest PRIVATE
    JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0)
target_link_libraries(StemForgeRoundtripTest
    PRIVATE
        juce::juce_audio_formats
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags)

add_test(NAME roundtrip_lossless
    COMMAND StemForgeRoundtripTest
            "${CMAKE_CURRENT_SOURCE_DIR}/Tests/fixtures/sine_stereo_f32.wav")

# Phase 1: overlap-add reconstruction must be lossless for an identity model (no ORT needed).
juce_add_console_app(StemForgeChunkerTest PRODUCT_NAME "StemForgeChunkerTest")
target_sources(StemForgeChunkerTest PRIVATE
    Tests/AudioChunkerTest.cpp
    Source/Separation/AudioChunker.cpp)
target_include_directories(StemForgeChunkerTest PRIVATE Source)
target_compile_definitions(StemForgeChunkerTest PRIVATE
    JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0)
target_link_libraries(StemForgeChunkerTest
    PRIVATE
        juce::juce_audio_basics
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags)

add_test(NAME chunker_overlap_add_lossless COMMAND StemForgeChunkerTest)
