#ifndef ONNXWORKER_H
#define ONNXWORKER_H

#include <string>
#include "onnxruntime_c_api.h"
#include <vector>
#include <utility>

class ONNXWorker
{
public:
    ONNXWorker(const std::string &modelPath);
    ~ONNXWorker();

    size_t getInputNodesNum();
    size_t getOutputNodesNum();
    std::vector<const char*> getInputNodesNames();
    std::vector<const char*> getOutputNodesNames();
    std::vector<ONNXTensorElementDataType> getInputNodesType();
    std::vector<ONNXTensorElementDataType> getOutputNodesType();
    std::vector<std::pair<size_t, std::vector<int64_t>>> getInputNodesDims();
    std::vector<std::pair<size_t, std::vector<int64_t>>> getOutputNodesDims();
    std::vector<size_t> getInputTensorSizes();

    std::vector<float> prepareSingleInputTensorData(size_t input_tensor_size);
    
    std::vector<float> getOutputDirect();
    bool CheckModelInfo();
private:
    bool CheckStatus(OrtStatus* status);


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