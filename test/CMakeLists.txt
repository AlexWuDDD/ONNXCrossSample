cmake_minimum_required(VERSION 3.13)
project(ONNXRuntimeTest)

set(CMAKE_CXX_STANDARD 11)

set(INC_DIR1 ${PROJECT_SOURCE_DIR}/include)
set(INC_DIR2 ${PROJECT_SOURCE_DIR}/include/onnxruntime)
set(INC_DIR3 ${PROJECT_SOURCE_DIR}/include/onnxruntime/core)
# set(INC_DIR4 ${PROJECT_SOURCE_DIR}/include/onnxruntime/core/common)
# set(INC_DIR5 ${PROJECT_SOURCE_DIR}/include/onnxruntime/core/framework)
# set(INC_DIR6 ${PROJECT_SOURCE_DIR}/include/onnxruntime/core/graph)
# set(INC_DIR7 ${PROJECT_SOURCE_DIR}/include/onnxruntime/core/optimizer)
# set(INC_DIR8 ${PROJECT_SOURCE_DIR}/include/onnxruntime/core/platform)
# set(INC_DIR9 ${PROJECT_SOURCE_DIR}/include/onnxruntime/core/providers)
set(INC_DIR10 ${PROJECT_SOURCE_DIR}/include/onnxruntime/core/session)

set(LINK_DIR /usr/IDAS/ONNX/libs)

include_directories(
      ${INC_DIR1} 
      ${INC_DIR2} 
      ${INC_DIR3} 
      # ${INC_DIR4} 
      # ${INC_DIR5} 
      # ${INC_DIR6} 
      # ${INC_DIR7}
      # ${INC_DIR8} 
      # ${INC_DIR9} 
      ${INC_DIR10})
link_directories(${LINK_DIR})

add_executable(testEndian ./src/testEndian.cpp)

add_executable(test1 ./src/test1.cpp)
target_link_libraries(test1 onnxruntime pthread atomic)


