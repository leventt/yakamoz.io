cmake_minimum_required(VERSION 3.12)
project(yakamozZFP)

set(CMAKE_TOOLCHAIN_FILE $EMSCRIPTEN/cmake/Modules/Platform/Emscripten.cmake)
set(CMAKE_BUILD_TYPE MinSizeRel)

set(BIT_STREAM_WORD_TYPE uint64)

add_subdirectory(zfp)

include_directories(${PROJECT_SOURCE_DIR}/zfp/include)
add_executable(zfpHelper zfpHelper.c)
target_link_libraries(zfpHelper zfp)

set_target_properties(
    zfpHelper PROPERTIES LINK_FLAGS
    "-s INITIAL_MEMORY=33554432 -s EXTRA_EXPORTED_RUNTIME_METHODS=['cwrap']"
)
