cmake_minimum_required(VERSION 3.12)
project(yakamozZFP)

set (CMAKE_CXX_STANDARD 14)
set(CMAKE_TOOLCHAIN_FILE $EMSCRIPTEN/cmake/Modules/Platform/Emscripten.cmake)
set(CMAKE_BUILD_TYPE Release)

set(BIT_STREAM_WORD_TYPE uint64)

add_subdirectory(zfp)

include_directories(${PROJECT_SOURCE_DIR}/zfp/include)
add_executable(zfpHelper zfpHelper.cxx)
target_link_libraries(zfpHelper zfp)

set_target_properties(
    zfpHelper PROPERTIES LINK_FLAGS
    "-O2 -s ALLOW_MEMORY_GROWTH=1 -s EXTRA_EXPORTED_RUNTIME_METHODS=\"['zfpHelper', '_malloc', '_free', 'cwrap', 'lengthBytesUTF8', 'stringToUTF8']\""
)
