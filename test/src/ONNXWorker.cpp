#include "ONNXWorker.h"
#include <cassert>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <random>

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
        printf("ONNXWorker::getInputNodesNum(): %d\n", num_input_nodes);
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
        printf("ONNXWorker::getOutputNodesNum(): %d\n", num_output_nodes);
        return num_output_nodes;
    }
    else{
        return -1;
    }
}

std::vector<const char*> ONNXWorker::getInputNodesNames(size_t input_nodes_size)
{
    std::vector<const char*> ret;
    for(int i = 0; i < input_nodes_size; ++i){
        char* input_name = nullptr;
        bool flag = CheckStatus(g_ort->SessionGetInputName(session, i, allocator, &input_name));
        if(flag){
            printf("ONNXWorker::getInputNodesNames() - get input node name: %s\n", input_name);
            ret.emplace_back(input_name);
        }
    }
    return ret;
}

std::vector<ONNXType> ONNXWorker::getInputNodesONNXType(size_t input_node_size)
{
    std::vector<ONNXType> ret;
    for(int i = 0; i < input_node_size; ++i){
        OrtTypeInfo* typeinfo;
        bool flag = CheckStatus(g_ort->SessionGetInputTypeInfo(session, i, &typeinfo));
        if(!flag){
            g_ort->ReleaseTypeInfo(typeinfo);
            continue;
        }
        ONNXType type;
        g_ort->GetOnnxTypeFromTypeInfo(typeinfo, &type);
        printf("ONNXWorker::getInputNodesONNXType() - ONNX type is : %d\n", type);
        ret.emplace_back(type);
        g_ort->ReleaseTypeInfo(typeinfo);
    }
    return ret;
}

ONNXTensorElementDataType ONNXWorker::getInputNodesElementDataType_ONNXType_Tensor(int index)
{
    OrtTypeInfo* typeinfo;
    bool flag = CheckStatus(g_ort->SessionGetInputTypeInfo(session, index, &typeinfo));
    if(!flag){
        g_ort->ReleaseTypeInfo(typeinfo);
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
    const OrtTensorTypeAndShapeInfo* tensor_info;
    flag = CheckStatus(g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
    if(!flag){
        // g_ort->ReleaseTensorTypeAndShapeInfo(tensor_info);
        g_ort->ReleaseTypeInfo(typeinfo);
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
    ONNXTensorElementDataType type;
    flag = CheckStatus(g_ort->GetTensorElementType(tensor_info, &type));
    if(!flag){
        // g_ort->ReleaseTensorTypeAndShapeInfo(tensor_info);
        g_ort->ReleaseTypeInfo(typeinfo);
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
    // g_ort->ReleaseTensorTypeAndShapeInfo(tensor_info);
    g_ort->ReleaseTypeInfo(typeinfo);
    printf("ONNXWorker::getInputNodesElementDataType_ONNXType_Tensor() - %d\n", type);
    return type;
}


std::vector<const char*> ONNXWorker::getOutputNodesNames(size_t output_nodes_size)
{
    std::vector<const char*> ret;
    for(int i = 0; i < output_nodes_size; ++i){
        char* output_name;
        bool flag = CheckStatus(g_ort->SessionGetOutputName(session, i, allocator, &output_name));
        if(flag){
            ret.emplace_back(output_name);
            printf("ONNXWorker::getOutputNodesNames() - get output node name: %s\n", output_name);
        }
    }
    return ret;    
}

std::vector<ONNXType> ONNXWorker::getOutputNodesONNXType(size_t output_node_size)
{
    std::vector<ONNXType> ret;
    for(int i = 0; i < output_node_size; ++i){
        OrtTypeInfo* typeinfo;
        bool flag = CheckStatus(g_ort->SessionGetOutputTypeInfo(session, i, &typeinfo));
        if(!flag){
            g_ort->ReleaseTypeInfo(typeinfo);
            continue;
        }
        ONNXType type;
        g_ort->GetOnnxTypeFromTypeInfo(typeinfo, &type);
        printf("ONNXWorker::getOutputNodesONNXType() - ONNX type is : %d\n", type);
        ret.emplace_back(type);
        g_ort->ReleaseTypeInfo(typeinfo);
    }
    return ret;
}

ONNXTensorElementDataType ONNXWorker::getOutputNodesElementDataType_ONNXType_Tensor(int index)
{
    OrtTypeInfo* typeinfo;
    bool flag = CheckStatus(g_ort->SessionGetOutputTypeInfo(session, index, &typeinfo));
    if(!flag){
        g_ort->ReleaseTypeInfo(typeinfo);
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
    const OrtTensorTypeAndShapeInfo* tensor_info;
    flag = CheckStatus(g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
    if(!flag){
        // g_ort->ReleaseTensorTypeAndShapeInfo(tensor_info);
        g_ort->ReleaseTypeInfo(typeinfo);
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
    ONNXTensorElementDataType type;
    flag = CheckStatus(g_ort->GetTensorElementType(tensor_info, &type));
    if(!flag){
        // g_ort->ReleaseTensorTypeAndShapeInfo(tensor_info);
        g_ort->ReleaseTypeInfo(typeinfo);
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
    // g_ort->ReleaseTensorTypeAndShapeInfo(tensor_info);
    g_ort->ReleaseTypeInfo(typeinfo);
    return type;
}

std::vector<std::pair<size_t, std::vector<int64_t>>> ONNXWorker::getInputNodesDims(size_t input_node_size)
{
    std::vector<std::pair<size_t, std::vector<int64_t>>> ret;
    for(int i = 0; i < input_node_size; ++i){
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
        printf("ONNXWorker::getInputNodesDims() - num_dims: %d\n", num_dims);
        for(const auto &item: input_node_dims){
            printf("%d\n", item);
        }
        g_ort->ReleaseTypeInfo(typeinfo);
    }
    return ret;
}

std::vector<std::pair<size_t, std::vector<int64_t>>> ONNXWorker::getOutputNodesDims(size_t output_nodes_size)
{
    std::vector<std::pair<size_t, std::vector<int64_t>>> ret;

    for(int i = 0; i < output_nodes_size; ++i){
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
        std::vector<int64_t> output_node_dims;
        output_node_dims.resize(num_dims);
        flag = CheckStatus(g_ort->GetDimensions(tensor_info, (int64_t*)output_node_dims.data(), num_dims));
        if(!flag){
            g_ort->ReleaseTypeInfo(typeinfo);
            continue;
        }
        ret.emplace_back(std::make_pair(num_dims, output_node_dims));
        printf("ONNXWorker::getOutputNodesDims() - num_dims: %d\n", num_dims);
        for(const auto &item: output_node_dims){
            printf("%d\n", item);
        }
        g_ort->ReleaseTypeInfo(typeinfo);
    }

    return ret;
}

std::vector<size_t> ONNXWorker::getInputTensorSizes(size_t input_node_size)
{
    std::vector<size_t> ret;

    for(int i = 0; i < input_node_size; ++i){
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
        printf("ONNXWorker::getInputTensorSizes: %d\n", input_tensor_size);
        g_ort->ReleaseTypeInfo(typeinfo);
    }

    return ret;
}


std::vector<size_t> ONNXWorker::getOutputTensorSizes(size_t output_node_size)
{
    std::vector<size_t> ret;

    for(int i = 0; i < output_node_size; ++i){
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
        size_t input_tensor_size = 0;

        flag = CheckStatus(g_ort->GetTensorShapeElementCount(tensor_info, &input_tensor_size));
        if(!flag || input_tensor_size <= 0){
            g_ort->ReleaseTypeInfo(typeinfo);
            continue;
        }
        ret.emplace_back(input_tensor_size);
        printf("ONNXWorker::getOutputTensorSizes: %d\n", input_tensor_size);
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

std::vector<float> ONNXWorker::prepareSingleInputTensorData2(size_t input_tensor_size)
{
    std::vector<float> input_tensor_values(input_tensor_size);
    float array[] = {29.25310295, 71.82661175, 26.44992148};

    for (size_t i = 0; i < input_tensor_size; i++){
        input_tensor_values[i] = array[i];
    }
    printf("data size: %d\n", input_tensor_values.size());
    return input_tensor_values;
}

std::vector<float> ONNXWorker::prepareSingleInputTensorData3(size_t input_tensor_size)
{
    std::vector<float> input_tensor_values(input_tensor_size);
    float array[] = {78.66414013, 14.7832246, 29.25310295, 71.82661175, 26.44992148 ,37.27810964, 71.55267399,
            27.32709227, 39.55344462, 13.16606454, 57.52491464
    };

    for (size_t i = 0; i < input_tensor_size; i++){
        int randonIndexArray = getRandomIndex(0, 10);
        printf("ONNXWorker::prepareSingleInputTensorData3() - input is %f\n", array[randonIndexArray]);
        input_tensor_values[i] = array[randonIndexArray];
    }
    printf("data size: %d\n", input_tensor_values.size());
    return input_tensor_values;
}

std::vector<float> ONNXWorker::getOutputDirect()
{
    printf("ONNXWorker::getOutputDirect()\n");
    int input_node_size = getInputNodesNum();
    int output_node_size = getOutputNodesNum();

    std::vector<float> ret;
    size_t input_tensor_size = getInputTensorSizes(input_node_size)[0];
    std::vector<float> input_tensor_values = prepareSingleInputTensorData(input_tensor_size);

    std::vector<const char*> input_nodesnames = getInputNodesNames(input_node_size);
    std::vector<std::pair<size_t, std::vector<int64_t>>> nodes_dims = getInputNodesDims(input_node_size);
    std::vector<size_t> input_tensor_sizes = getInputTensorSizes(input_node_size);
    ONNXTensorElementDataType input_tensor_types = getInputNodesElementDataType_ONNXType_Tensor(0);

    OrtMemoryInfo* memory_info;
    CheckStatus(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    OrtValue* input_tensor = NULL;
    CheckStatus(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values.data(), input_tensor_size * sizeof(float), nodes_dims[0].second.data(), nodes_dims[0].second.size(), input_tensor_types, &input_tensor));
    int is_tensor;
    CheckStatus(g_ort->IsTensor(input_tensor, &is_tensor));
    assert(is_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
    // score model & input tensor, get back output tensor
    OrtValue* output_tensor = NULL;
    std::vector<const char*> input_node_names = getInputNodesNames(input_node_size);
    std::vector<const char*> output_node_names = getOutputNodesNames(output_node_size);
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

/******************************************************************************************************/

bool ONNXWorker::getInputsInfo(std::vector<IOInfo> &rets)
{
    //get input nodes num
    size_t input_nodes_num = getInputNodesNum();
    if(input_nodes_num <= 0){
        printf("ONNXWorker::getInputsInfo() - getInputNodesNum - ERROR\n");
        return false;
    }

    printf("ONNXWorker::getInputsInfo() - we have %d input nodes\n", input_nodes_num);

    //get input nodes name
    std::vector<const char*> input_nodes_names = getInputNodesNames(input_nodes_num);
    if(input_nodes_names.size() != input_nodes_num){
        printf("ONNXWorker::getInputsInfo() - get InputNodesName - ERROR\n");
        return false;
    }

    //get input nodes ONNXType
    std::vector<ONNXType> input_nodes_ONNXTypes = getInputNodesONNXType(input_nodes_num);
    if(input_nodes_ONNXTypes.size() != input_nodes_num){
        printf("ONNXWorker::getInputsInfo() - get InputNodes ONNXType - ERROR\n");
        return false;
    }

    std::vector<ONNXTensorElementDataType> input_ElementData_types;
    for(int i = 0 ; i < input_nodes_ONNXTypes.size(); ++i){
        switch (input_nodes_ONNXTypes[i]){
            case ONNXType::ONNX_TYPE_TENSOR:
                input_ElementData_types.emplace_back(getInputNodesElementDataType_ONNXType_Tensor(i));
                break;
            default:
                printf("ONNXWorker::getInputsInfo() - unsupported ONNX TYPE currently");
                return false;
        }
    }

    std::vector<std::pair<size_t, std::vector<int64_t>>> intput_nodes_dims = getInputNodesDims(input_nodes_num);
    if(intput_nodes_dims.size() != input_nodes_num){
        printf("ONNXWorker::getInputsInfo() - get InputNodes Dims - ERROR\n");
        return false;
    }

    std::vector<size_t> input_nodes_tensors_size = getInputTensorSizes(input_nodes_num);

    return true;
}

bool ONNXWorker::getOutputsInfo(std::vector<IOInfo> &rets)
{
    //get output nodes num
    size_t output_nodes_num = getOutputNodesNum();
    if(output_nodes_num <= 0){
        printf("ONNXWorker::getOutputsInfo() - getOutputNodesNum - ERROR\n");
        return false;
    }

    printf("ONNXWorker::getOutputsInfo() - we have %d output nodes\n", output_nodes_num);

    //get output nodes name
    std::vector<const char*> output_nodes_names = getOutputNodesNames(output_nodes_num);
    if(output_nodes_names.size() != output_nodes_num){
        printf("ONNXWorker::getOutputsInfo() - get OutputNodesName - ERROR\n");
        return false;
    }

    //get input nodes ONNXType
    std::vector<ONNXType> output_nodes_ONNXTypes = getOutputNodesONNXType(output_nodes_num);
    if(output_nodes_ONNXTypes.size() != output_nodes_num){
        printf("ONNXWorker::getOutputsInfo() - get OutputNodes ONNXType - ERROR\n");
        return false;
    }

    std::vector<ONNXTensorElementDataType> output_ElementData_types;
    for(int i = 0 ; i < output_nodes_ONNXTypes.size(); ++i){
        switch (output_nodes_ONNXTypes[i]){
            case ONNXType::ONNX_TYPE_TENSOR:
                output_ElementData_types.emplace_back(getOutputNodesElementDataType_ONNXType_Tensor(i));
                break;
            default:
                printf("ONNXWorker::getOutputsInfo() - unsupported ONNX TYPE currently");
                return false;
        }
    }

    std::vector<std::pair<size_t, std::vector<int64_t>>> output_nodes_dims = getOutputNodesDims(output_nodes_num);
    if(output_nodes_dims.size() != output_nodes_num){
        printf("ONNXWorker::getOutputsInfo() - get OutputNodes Dims - ERROR\n");
        return false;
    }

    std::vector<size_t> output_nodes_tensors_size = getOutputTensorSizes(output_nodes_num);

    return true;
}

std::vector<float> ONNXWorker::getOutputDirect2()
{
    printf("ONNXWorker::getOutputDirect2()\n");
    int input_node_size = getInputNodesNum();
    int output_node_size = getOutputNodesNum();

    std::vector<float> ret;
    size_t input_tensor_size = 1;
    std::vector<float> input_tensor_values = prepareSingleInputTensorData2(input_tensor_size);

    std::vector<const char*> input_nodesnames = getInputNodesNames(input_node_size);
    std::vector<std::pair<size_t, std::vector<int64_t>>> nodes_dims = getInputNodesDims(input_node_size);
    // std::vector<size_t> input_tensor_sizes = getInputTensorSizes(input_node_size);
    ONNXTensorElementDataType input_tensor_types = getInputNodesElementDataType_ONNXType_Tensor(0);

    OrtMemoryInfo* memory_info;
    CheckStatus(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    OrtValue* input_tensor = NULL;
    CheckStatus(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values.data(), input_tensor_size * sizeof(float), nodes_dims[0].second.data(), nodes_dims[0].second.size(), input_tensor_types, &input_tensor));
    int is_tensor;
    CheckStatus(g_ort->IsTensor(input_tensor, &is_tensor));
    assert(is_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
    // score model & input tensor, get back output tensor
    OrtValue* output_tensor = NULL;
    std::vector<const char*> input_node_names = getInputNodesNames(input_node_size);
    std::vector<const char*> output_node_names = getOutputNodesNames(output_node_size);
    CheckStatus(g_ort->Run(session, NULL, input_node_names.data(), (const OrtValue* const*)&input_tensor, 1, output_node_names.data(), 1, &output_tensor));
    CheckStatus(g_ort->IsTensor(output_tensor, &is_tensor));
    assert(is_tensor);

    // Get pointer to output tensor float values
    float* floatarr;
    CheckStatus(g_ort->GetTensorMutableData(output_tensor, (void**)&floatarr));
    // assert(std::abs(floatarr[0] - 0.000045) < 1e-6);

    // score the model, and print scores for first 5 classes
    for (int i = 0; i < 1; i++){
        ret.emplace_back(floatarr[i]);
    }
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseValue(input_tensor);

    return ret;
}

std::vector<float> ONNXWorker::getOutputDirect3()
{
    printf("ONNXWorker::getOutputDirect3()\n");
    int input_node_size = getInputNodesNum();
    int output_node_size = getOutputNodesNum();

    std::vector<float> ret;
    size_t input_tensor_size = getInputTensorSizes(input_node_size)[0];
    std::vector<float> input_tensor_values = prepareSingleInputTensorData3(input_tensor_size);

    std::vector<const char*> input_nodesnames = getInputNodesNames(input_node_size);
    std::vector<std::pair<size_t, std::vector<int64_t>>> nodes_dims = getInputNodesDims(input_node_size);
    // std::vector<size_t> input_tensor_sizes = getInputTensorSizes(input_node_size);
    ONNXTensorElementDataType input_tensor_types = getInputNodesElementDataType_ONNXType_Tensor(0);

    OrtMemoryInfo* memory_info;
    CheckStatus(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    OrtValue* input_tensor = NULL;
    CheckStatus(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values.data(), input_tensor_size * sizeof(float), nodes_dims[0].second.data(), nodes_dims[0].second.size(), input_tensor_types, &input_tensor));
    int is_tensor;
    CheckStatus(g_ort->IsTensor(input_tensor, &is_tensor));
    assert(is_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
    // score model & input tensor, get back output tensor
    OrtValue* output_tensor = NULL;
    std::vector<const char*> input_node_names = getInputNodesNames(input_node_size);
    std::vector<const char*> output_node_names = getOutputNodesNames(output_node_size);
    CheckStatus(g_ort->Run(session, NULL, input_node_names.data(), (const OrtValue* const*)&input_tensor, 1, output_node_names.data(), 1, &output_tensor));
    CheckStatus(g_ort->IsTensor(output_tensor, &is_tensor));
    assert(is_tensor);

    // Get pointer to output tensor float values
    float* floatarr;
    CheckStatus(g_ort->GetTensorMutableData(output_tensor, (void**)&floatarr));
    // assert(std::abs(floatarr[0] - 0.000045) < 1e-6);

    // score the model, and print scores for first 5 classes
    for (int i = 0; i < 1; i++){
        ret.emplace_back(floatarr[i]);
    }
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseValue(input_tensor);

    return ret;
}

int ONNXWorker::getRandomIndex(int from, int end)
{
    // time_t seed = time(nullptr);
    // std::default_random_engine generator(seed);
    // std::uniform_int_distribution<int> distribution(from, end);
    // return distribution(generator);
    srand(time(nullptr));
    return rand()%10;
}