#include "ONNXWorker.h"
#include <cassert>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>

bool ONNXWorker::CheckStatus(OrtStatus* status)
{
    if (status != NULL) {
      const char* msg = g_ort->GetErrorMessage(status);
      fprintf(stderr, "%s\n", msg);
      g_ort->ReleaseStatus(status);
      return false;
    }
    return true;
}

ONNXWorker::ONNXWorker(const std::string &modelPath)
    :   g_ort(OrtGetApiBase()->GetApi(ORT_API_VERSION)), 
        model_path(modelPath),
        input_tensors_len(0)
{
    assert(g_ort != nullptr);
    bool ret = CheckStatus(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ONNXWorker", &env));
    assert(ret != false && env != nullptr);
    ret = CheckStatus(g_ort->CreateSessionOptions(&session_options));
    assert(ret != false && session_options != nullptr);
    ret = CheckStatus(g_ort->SetIntraOpNumThreads(session_options, 1));
    assert(ret != false);
    ret = CheckStatus(g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC));
    assert(ret != false);
    ret = CheckStatus(g_ort->CreateSession(env, model_path.c_str(), session_options, &session));
    assert(ret != false && session != nullptr);
    ret = CheckStatus(g_ort->GetAllocatorWithDefaultOptions(&allocator));
    assert(ret != false && allocator != nullptr);
}

ONNXWorker::~ONNXWorker()
{
  g_ort->ReleaseSession(session);
  g_ort->ReleaseSessionOptions(session_options);
  g_ort->ReleaseEnv(env);
}

size_t ONNXWorker::getInputNodesNum()
{
    size_t num_input_nodes = 0;
    bool ret = CheckStatus(g_ort->SessionGetInputCount(session, &num_input_nodes));
    if(ret){
        return num_input_nodes;
    }
    else{
        return -1;
    }
}

size_t ONNXWorker::getOutputNodesNum()
{
    size_t num_output_nodes = 0;
    bool ret = CheckStatus(g_ort->SessionGetOutputCount(session, &num_output_nodes));
    if(ret){
        return num_output_nodes;
    }
    else{
        return -1;
    }
}

std::vector<const char*> ONNXWorker::getInputNodesNames()
{
    std::vector<const char*> ret;
    size_t nums = getInputNodesNum();
    if(nums <= 0){
        return ret;
    }

    for(int i = 0; i < nums; ++i){
        char* input_name;
        bool flag = CheckStatus(g_ort->SessionGetInputName(session, i, allocator, &input_name));
        if(flag){
            ret.emplace_back(input_name);
        }
    }
    return ret;
}

std::vector<const char*> ONNXWorker::getOutputNodesNames()
{
    std::vector<const char*> ret;
    size_t nums = getOutputNodesNum();
    if(nums <= 0){
        return ret;
    }

    for(int i = 0; i < nums; ++i){
        char* input_name;
        bool flag = CheckStatus(g_ort->SessionGetOutputName(session, i, allocator, &input_name));
        if(flag){
            ret.emplace_back(input_name);
        }
    }
    return ret;    
}

std::vector<ONNXTensorElementDataType> ONNXWorker::getInputNodesType()
{
    std::vector<ONNXTensorElementDataType> ret;
    size_t nums = getInputNodesNum();
    if(nums <= 0){
        return ret;
    }
    for(int i = 0; i < nums; ++i){
        OrtTypeInfo* typeinfo;
        bool flag = CheckStatus(g_ort->SessionGetInputTypeInfo(session, i, &typeinfo));
        if(!flag){
            g_ort->ReleaseTypeInfo(typeinfo);
            continue;
        }
        const OrtTensorTypeAndShapeInfo* tensor_info;
        flag = CheckStatus(g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
        if(!flag){
            g_ort->ReleaseTypeInfo(typeinfo);
            continue;
        }
        ONNXTensorElementDataType type;
        flag = CheckStatus(g_ort->GetTensorElementType(tensor_info, &type));
        if(!flag){
            g_ort->ReleaseTypeInfo(typeinfo);
            continue;
        }
        ret.emplace_back(type);
        g_ort->ReleaseTypeInfo(typeinfo);
    }
    return ret;
}

std::vector<ONNXTensorElementDataType> ONNXWorker::getOutputNodesType()
{
    std::vector<ONNXTensorElementDataType> ret;
    size_t nums = getOutputNodesNum();
    if(nums <= 0){
        return ret;
    }
    for(int i = 0; i < nums; ++i){
        OrtTypeInfo* typeinfo;
        bool flag = CheckStatus(g_ort->SessionGetOutputTypeInfo(session, i, &typeinfo));
        if(!flag){
            g_ort->ReleaseTypeInfo(typeinfo);
            continue;
        }
        const OrtTensorTypeAndShapeInfo* tensor_info;
        flag = CheckStatus(g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
        if(!flag){
            g_ort->ReleaseTypeInfo(typeinfo);
            continue;
        }
        ONNXTensorElementDataType type;
        flag = CheckStatus(g_ort->GetTensorElementType(tensor_info, &type));
        if(!flag){
            g_ort->ReleaseTypeInfo(typeinfo);
            continue;
        }
        ret.emplace_back(type);
        g_ort->ReleaseTypeInfo(typeinfo);
    }
    return ret;    
}


std::vector<std::pair<size_t, std::vector<int64_t>>> ONNXWorker::getInputNodesDims()
{
    std::vector<std::pair<size_t, std::vector<int64_t>>> ret;
    size_t nums = getInputNodesNum();
    if(nums <= 0){
        return ret;
    }

    for(int i = 0; i < nums; ++i){
        OrtTypeInfo* typeinfo;
        bool flag = CheckStatus(g_ort->SessionGetInputTypeInfo(session, i, &typeinfo));
        if(!flag){
            g_ort->ReleaseTypeInfo(typeinfo);
            continue;
        }
        const OrtTensorTypeAndShapeInfo* tensor_info;
        flag = CheckStatus(g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
        if(!flag){
            g_ort->ReleaseTypeInfo(typeinfo);
            continue;
        }
        size_t num_dims = 0;
	    flag = CheckStatus(g_ort->GetDimensionsCount(tensor_info, &num_dims));
        if(!flag || num_dims <= 0){
            g_ort->ReleaseTypeInfo(typeinfo);
            continue;
        }
        std::vector<int64_t> input_node_dims;
        input_node_dims.resize(num_dims);
        flag = CheckStatus(g_ort->GetDimensions(tensor_info, (int64_t*)input_node_dims.data(), num_dims));
        if(!flag){
            g_ort->ReleaseTypeInfo(typeinfo);
            continue;
        }
        ret.emplace_back(std::make_pair(num_dims, input_node_dims));
        g_ort->ReleaseTypeInfo(typeinfo);
    }

    return ret;
}

std::vector<std::pair<size_t, std::vector<int64_t>>> ONNXWorker::getOutputNodesDims()
{
    std::vector<std::pair<size_t, std::vector<int64_t>>> ret;
    size_t nums = getOutputNodesNum();
    if(nums <= 0){
        return ret;
    }

    for(int i = 0; i < nums; ++i){
        OrtTypeInfo* typeinfo;
        bool flag = CheckStatus(g_ort->SessionGetOutputTypeInfo(session, i, &typeinfo));
        if(!flag){
            g_ort->ReleaseTypeInfo(typeinfo);
            continue;
        }
        const OrtTensorTypeAndShapeInfo* tensor_info;
        flag = CheckStatus(g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
        if(!flag){
            g_ort->ReleaseTypeInfo(typeinfo);
            continue;
        }
        size_t num_dims = 0;
	    flag = CheckStatus(g_ort->GetDimensionsCount(tensor_info, &num_dims));
        if(!flag || num_dims <= 0){
            g_ort->ReleaseTypeInfo(typeinfo);
            continue;
        }
        std::vector<int64_t> input_node_dims;
        input_node_dims.resize(num_dims);
        flag = CheckStatus(g_ort->GetDimensions(tensor_info, (int64_t*)input_node_dims.data(), num_dims));
        if(!flag){
            g_ort->ReleaseTypeInfo(typeinfo);
            continue;
        }
        ret.emplace_back(std::make_pair(num_dims, input_node_dims));
        g_ort->ReleaseTypeInfo(typeinfo);
    }

    return ret;
}

std::vector<size_t> ONNXWorker::getInputTensorSizes()
{
    std::vector<size_t> ret;
    size_t nums = getInputNodesNum();
    if(nums <= 0){
        return ret;
    }

    for(int i = 0; i < nums; ++i){
        OrtTypeInfo* typeinfo;
        bool flag = CheckStatus(g_ort->SessionGetInputTypeInfo(session, i, &typeinfo));
        if(!flag){
            g_ort->ReleaseTypeInfo(typeinfo);
            continue;
        }
        const OrtTensorTypeAndShapeInfo* tensor_info;
        flag = CheckStatus(g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
        if(!flag){
            g_ort->ReleaseTypeInfo(typeinfo);
            continue;
        }
        size_t num_dims = 0;
	    flag = CheckStatus(g_ort->GetDimensionsCount(tensor_info, &num_dims));
        if(!flag || num_dims <= 0){
            g_ort->ReleaseTypeInfo(typeinfo);
            continue;
        }
        size_t input_tensor_size = 0;

        flag = CheckStatus(g_ort->GetTensorShapeElementCount(tensor_info, &input_tensor_size));
        if(!flag || input_tensor_size <= 0){
            g_ort->ReleaseTypeInfo(typeinfo);
            continue;
        }
        ret.emplace_back(input_tensor_size);
        g_ort->ReleaseTypeInfo(typeinfo);
    }

    return ret;
}

std::vector<float> ONNXWorker::prepareSingleInputTensorData(size_t input_tensor_size)
{
    std::vector<float> input_tensor_values(input_tensor_size);

    for (size_t i = 0; i < input_tensor_size; i++){
        input_tensor_values[i] = (float)i / (input_tensor_size + 1);
    }
    printf("data size: %d\n", input_tensor_values.size());
    return input_tensor_values;
}

bool ONNXWorker::CheckModelInfo()
{
    printf("****************************************\n");
    getONNXTypeInfo();
    printf("****************************************\n");
    getOutputNodesType_ONNXTYPE_IS_SEQUENCE();
    printf("****************************************\n");
    
    size_t inputNodesNum = getInputNodesNum();
    printf("inputNodesNum: %d\n", inputNodesNum);
    std::vector<const char*> inputNodesNames = getInputNodesNames();
    std::vector<size_t> input_tensor_sizes = getInputTensorSizes();


    printf("inputNodesNames:\n");
    for(const auto &item : inputNodesNames){
        printf("%s\n", item);
    }

    std::vector<std::pair<size_t, std::vector<int64_t>>> inputNodesDims = getInputNodesDims();
    if(inputNodesNum == inputNodesNames.size() && inputNodesNum == inputNodesDims.size() && inputNodesNum == input_tensor_sizes.size()){
        for(int i = 0; i < inputNodesNum; ++i){
            if(inputNodesDims[i].first != inputNodesDims[i].second.size()){
                printf("input nodes: %s --- dims issue: should %d but %d", inputNodesDims[i].first, inputNodesDims[i].second.size());
                return false;
            }
        }
    }
    else{
        printf("input nodes num : %d ,nodes name size: %d, dims size: %d", inputNodesNum, inputNodesNames.size(), inputNodesDims.size());
        return false;
    }
    printf("inputNodesDims:\n");
    for(const auto &item : inputNodesDims){
        printf("%d: ", item.first);
        for(const auto &dim: item.second){
            printf("%d ", dim);
        }
        printf("\n");
    }

    printf("input tensor sizes:\n");
    for(const auto &item : input_tensor_sizes){
        printf("%d ", item);
    }
    printf("\n");

    std::vector<ONNXTensorElementDataType> input_types = getInputNodesType();
    printf("input node types:\n");
    for (const auto &item: input_types)
    {
        printf("%d ", item);
    }
    printf("\n");
    
    std::vector<ONNXTensorElementDataType> output_types = getOutputNodesType();
    printf("output node types:\n");
    for (const auto &item: output_types)
    {
        printf("%d ", item);
    }
    printf("\n");

    size_t outputNodesNum = getOutputNodesNum();
    printf("outputNodesNum: %d\n", outputNodesNum);
    std::vector<const char*> outputNodesNames = getOutputNodesNames();
    std::vector<std::pair<size_t, std::vector<int64_t>>> outputNodesDims = getOutputNodesDims();

    if(outputNodesNum != outputNodesNames.size() || outputNodesNum != outputNodesDims.size()){
        printf("output nodes num : %d ,nodes name size: %d, nodes dims size: %d", outputNodesNum, outputNodesNames.size(), outputNodesDims.size());
        return false;
    }
    printf("outputNodesNames:\n");
    for(const auto &item : outputNodesNames){
        printf("%s\n", item);
    }

    input_tensors_len = inputNodesNum;
    printf("input_tensors_len: %d\n", input_tensors_len);

    printf("outputNodesDims:\n");
    for(const auto &item : outputNodesDims){
        printf("%d: ", item.first);
        for(const auto &dim: item.second){
            printf("%d ", dim);
        }
        printf("\n");
    }

    return true;
}


std::vector<float> ONNXWorker::getOutputDirect()
{
    printf("ONNXWorker::getOutputDirect()\n");
    std::vector<float> ret;
    size_t input_tensor_size = getInputTensorSizes()[0];
    std::vector<float> input_tensor_values = prepareSingleInputTensorData(input_tensor_size);

    std::vector<const char*> input_nodesnames = getInputNodesNames();
    std::vector<std::pair<size_t, std::vector<int64_t>>> nodes_dims = getInputNodesDims();
    std::vector<size_t> input_tensor_sizes = getInputTensorSizes();
    std::vector<ONNXTensorElementDataType> input_tensor_types = getInputNodesType();

    OrtMemoryInfo* memory_info;
    CheckStatus(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    OrtValue* input_tensor = NULL;
    CheckStatus(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values.data(), input_tensor_size * sizeof(float), nodes_dims[0].second.data(), nodes_dims[0].second.size(), input_tensor_types[0], &input_tensor));
    int is_tensor;
    CheckStatus(g_ort->IsTensor(input_tensor, &is_tensor));
    assert(is_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
    // score model & input tensor, get back output tensor
    OrtValue* output_tensor = NULL;
    std::vector<const char*> input_node_names = getInputNodesNames();
    std::vector<const char*> output_node_names = getOutputNodesNames();
    CheckStatus(g_ort->Run(session, NULL, input_node_names.data(), (const OrtValue* const*)&input_tensor, 1, output_node_names.data(), 1, &output_tensor));
    CheckStatus(g_ort->IsTensor(output_tensor, &is_tensor));
    assert(is_tensor);

    // Get pointer to output tensor float values
    float* floatarr;
    CheckStatus(g_ort->GetTensorMutableData(output_tensor, (void**)&floatarr));
    // assert(std::abs(floatarr[0] - 0.000045) < 1e-6);

    // score the model, and print scores for first 5 classes
    for (int i = 0; i < 10; i++){
        ret.emplace_back(floatarr[i]);
    }
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseValue(input_tensor);

    return ret;
}
/************************************************************************************************************/
void ONNXWorker::getONNXTypeInfo()
{
    std::vector<ONNXTensorElementDataType> ret;
    size_t nums = getInputNodesNum();
    if(nums <= 0){
        printf ("ONNXWorker::getONNXTypeInfo() - 1");
        return ;
    }
    for(int i = 0; i < nums; ++i){
        OrtTypeInfo* typeinfo;
        bool flag = CheckStatus(g_ort->SessionGetInputTypeInfo(session, i, &typeinfo));
        if(!flag){
            g_ort->ReleaseTypeInfo(typeinfo);
            continue;
        }
        ONNXType type;
        g_ort->GetOnnxTypeFromTypeInfo(typeinfo, &type);
        printf("InputNode type is : %d\n", type);
        g_ort->ReleaseTypeInfo(typeinfo);
    }

    nums = getOutputNodesNum();
    if(nums <= 0){
        printf ("ONNXWorker::getONNXTypeInfo() - 1");
        return;
    }
    for(int i = 0; i < nums; ++i){
        OrtTypeInfo* typeinfo;
        bool flag = CheckStatus(g_ort->SessionGetOutputTypeInfo(session, i, &typeinfo));
        if(!flag){
            g_ort->ReleaseTypeInfo(typeinfo);
            continue;
        }
        ONNXType type;
        g_ort->GetOnnxTypeFromTypeInfo(typeinfo, &type);
        printf("OutputNode type is : %d\n", type);
        g_ort->ReleaseTypeInfo(typeinfo);
    }
    return ;
}

void ONNXWorker::getOutputNodesType_ONNXTYPE_IS_SEQUENCE()
{
    size_t nums = getOutputNodesNum();
    if(nums <= 0){
        return;
    }
    for(int i = 0; i < nums; ++i){
        OrtTypeInfo* typeinfo;
        bool flag = CheckStatus(g_ort->SessionGetOutputTypeInfo(session, i, &typeinfo));
        if(!flag){
            g_ort->ReleaseTypeInfo(typeinfo);
            continue;
        }
        ONNXType type;
        g_ort->GetOnnxTypeFromTypeInfo(typeinfo, &type);
        if(type == ONNXType::ONNX_TYPE_TENSOR){
            const OrtTensorTypeAndShapeInfo* tensor_info;
            flag = CheckStatus(g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
            if(!flag){
                g_ort->ReleaseTypeInfo(typeinfo);
                continue;
            }
            ONNXTensorElementDataType type;
            flag = CheckStatus(g_ort->GetTensorElementType(tensor_info, &type));
            if(!flag){
                g_ort->ReleaseTypeInfo(typeinfo);
                continue;
            }
            printf("tensor nodes type: %d\n", type);
        }
        else if(type == ONNXType::ONNX_TYPE_SEQUENCE){
            const OrtSequenceTypeInfo *sequence_info;
            flag = CheckStatus(g_ort->CastTypeInfoToSequenceTypeInfo(typeinfo, &sequence_info));
            if(!flag){
                g_ort->ReleaseTypeInfo(typeinfo);
                continue;
            }
            OrtTypeInfo *type;
            flag = CheckStatus(g_ort->GetSequenceElementType(sequence_info, &type));
            if(!flag){
                g_ort->ReleaseTypeInfo(typeinfo);
                g_ort->ReleaseTypeInfo(type);
                continue;
            }
            ONNXType otype;
            g_ort->GetOnnxTypeFromTypeInfo(type, &otype);
            printf("sequecne item ONNXType: %d\n", otype);
            if(otype == ONNXType::ONNX_TYPE_MAP){
                const OrtMapTypeInfo * map_info;
                g_ort->CastTypeInfoToMapTypeInfo(type, &map_info);
                ONNXTensorElementDataType keytype;
                g_ort->GetMapKeyType(map_info, &keytype);
                ONNXTensorElementDataType valuetype;
                g_ort->GetMapKeyType(map_info, &valuetype);

                printf("in sequecne: key is %d, value is %d\n", keytype, valuetype);
            }


            g_ort->ReleaseTypeInfo(type);
        }
        g_ort->ReleaseTypeInfo(typeinfo);
    }
}