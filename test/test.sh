#! /bin/bash

M881_ToolChain_Root="/home/alexdockermother/Documents/m881/aarch64/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin"
MYCXX="${M881_ToolChain_Root}/aarch64-linux-gnu-g++"
MYCC="${M881_ToolChain_Root}/aarch64-linux-gnu-gcc"

DIR=`pwd`

cd build
rm -rf *
cd $DIR

cmake -S "." -B "./build" -DCMAKE_C_COMPILER="${MYCC}" -DCMAKE_CXX_COMPILER="${MYCXX}"
make -C "./build"

cp ./build/test* ./bin/


BoxIP="10.0.0.197"
User="root"
SourceFolder1="./bin"
# SourceFolder2="./libs"
SourceFolder3="./model"
TargetFolder="/usr/IDAS/ONNX"

echo "This foler - ${SourceFolder1} - is upload the newest file to the target box - ${TargetFolder}"
# echo "This foler - ${SourceFolder2} - is upload the newest file to the target box - ${TargetFolder}"
echo "This foler - ${SourceFolder3} - is upload the newest file to the target box - ${TargetFolder}"


ssh "${User}@${BoxIP}" "[ -d ${TargetFolder} ] && echo ok || mkdir -p ${TargetFolder}"
scp -prq "${SourceFolder1}" "${User}@${BoxIP}:${TargetFolder}"
# scp -prq "${SourceFolder2}" "${User}@${BoxIP}:${TargetFolder}"
scp -prq "${SourceFolder3}" "${User}@${BoxIP}:${TargetFolder}"