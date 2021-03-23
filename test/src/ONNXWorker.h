#ifndef ONNXWORKER_H
#define ONNXWORKER_H

#include <string>
#include "onnxruntime_c_api.h"
#include <vector>
#include <utility>

struct IOInfo{
    std::string name;
    ONNXTensorElementDataType datatype;
    std::pair<size_t, std::vector<int64_t>> Dims;
    size_t DataNums;
};


class ONNXWorker
{
public:
    ONNXWorker(const std::string &modelPath);
    ~ONNXWorker();

    bool getInputsInfo(std::vector<IOInfo> &rets);
    bool getOutputsInfo(std::vector<IOInfo> &rets);
    
    std::vector<float> getOutputDirect();
    std::vector<float> getOutputDirect2();
    std::vector<float> getOutputDirect3();
private:
    size_t getInputNodesNum();
    size_t getOutputNodesNum();

    std::vector<const char*> getInputNodesNames(size_t input_nodes_size);
    std::vector<ONNXType> getInputNodesONNXType(size_t input_node_size);
    ONNXTensorElementDataType getInputNodesElementDataType_ONNXType_Tensor(int index);
    std::vector<std::pair<size_t, std::vector<int64_t>>> getInputNodesDims(size_t input_node_size);
    std::vector<size_t> getInputTensorSizes(size_t input_node_size);

    std::vector<const char*> getOutputNodesNames(size_t output_nodes_size);
    std::vector<ONNXType> getOutputNodesONNXType(size_t output_nodes_size);
    ONNXTensorElementDataType getOutputNodesElementDataType_ONNXType_Tensor(int index);
    std::vector<std::pair<size_t, std::vector<int64_t>>> getOutputNodesDims(size_t output_nodes_size);
    std::vector<size_t> getOutputTensorSizes(size_t output_node_size);

    std::vector<float> prepareSingleInputTensorData(size_t input_tensor_size);
    std::vector<float> prepareSingleInputTensorData2(size_t input_tensor_size);
    std::vector<float> prepareSingleInputTensorData3(size_t input_tensor_size);


    void getONNXTypeInfo();
    // std::vector<ONNXTensorElementDataType> getOutputNodesType_ONNXTYPE_IS_TENSOR();
    void getOutputNodesType_ONNXTYPE_IS_SEQUENCE();

private:
    bool CheckStatus(OrtStatus* status);
    int getRandomIndex(int from, int end);

private:
    const OrtApi* g_ort;
    OrtEnv* env;
    OrtSessionOptions* session_options;
    OrtSession* session;
    std::string model_path;
    OrtAllocator* allocator;

    int input_tensors_len;
    
    std::vector<OrtValue*> input_tensors;


    ONNXTensorElementDataType datatype;
};
#endif