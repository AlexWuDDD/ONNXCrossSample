#include "ONNXWorker.h"
#include <stdio.h>
#include <unistd.h>

#define MODEL_PATH_1 "/usr/IDAS/ONNX/model/squeezenet.onnx"
#define MODEL_PATH_2 "/usr/IDAS/ONNX/model/logreg_iris.onnx"
#define MODEL_PATH_3 "/usr/IDAS/ONNX/model/super_resolution.onnx"
#define MODEL_PATH_4 "/usr/IDAS/ONNX/model/mlp.onnx"
#define MODEL_PATH_5 "/usr/IDAS/ONNX/model/easy_example.onnx"
#define MODEL_PATH_5 "/usr/IDAS/ONNX/model/easy_example_2.onnx"

int main(int argc, char const *argv[])
{
    ONNXWorker *worker = new ONNXWorker(MODEL_PATH_5);
    std::vector<IOInfo> rets;
    bool flag1 = worker->getInputsInfo(rets);
    bool flag2 = worker->getOutputsInfo(rets);
    printf("flag1: %d - flag2: %d\n", flag1, flag2);

    if(flag1 && flag2){
        printf("Check model info pass !!!\n");
        // auto result = worker->getOutputDirect();
        for(int i = 0; i < 10; ++i){
            auto result = worker->getOutputDirect3();
            if(result.empty()){
                printf("OUTPUT ERROR\n");
            }
            else{
                printf("result:\n");
                for(const auto & value: result){
                    printf("%f ", value);
                }
                printf("\n");
            }
            sleep(1);
        }
    }
    else{
        printf("Check model info fail !!!\n");
    }

    delete worker;
    return 0;
}
