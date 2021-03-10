#! /bin/bash

cd onnxruntime

DIR=`pwd`
# Download protoc if we don't have it
if [ ! -d "build/protoc" ]; then
	mkdir -p "build/protoc"
	curl --location "https://github.com/protocolbuffers/protobuf/releases/download/v3.11.3/protoc-3.11.3-linux-x86_64.zip" --output "build/protoc/protoc.zip"
	cd build/protoc
	unzip protoc.zip
fi

cd $DIR

M881_ToolChain_Root="/home/alexdockermother/Documents/m881/aarch64/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin"
MYCXX="${M881_ToolChain_Root}/aarch64-linux-gnu-g++"
MYCC="${M881_ToolChain_Root}/aarch64-linux-gnu-gcc"
PROTOC="/home/alexdockermother/Documents/MyProject/ONNX/onnxruntime/build/protoc/bin/protoc"
./build.sh --config Release --parallel --build_dir=build --build_shared_lib --arm64 --skip_tests --cmake_extra_defines CMAKE_C_COMPILER="${MYCC}" CMAKE_CXX_COMPILER="${MYCXX}" ONNX_CUSTOM_PROTOC_EXECUTABLE=$PROTOC