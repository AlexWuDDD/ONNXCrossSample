#include "ONNXWorker.h"
#include <stdio.h>

#define MODEL_PATH_1 "/usr/IDAS/ONNX/model/squeezenet.onnx"
#define MODEL_PATH_2 "/usr/IDAS/ONNX/model/logreg_iris.onnx"
#define MODEL_PATH_3 "/usr/IDAS/ONNX/model/super_resolution.onnx"
#define MODEL_PATH_4 "/usr/IDAS/ONNX/model/mlp.onnx"

int main(int argc, char const *argv[])
{
    ONNXWorker *worker = new ONNXWorker(MODEL_PATH_4);
    if(worker->CheckModelInfo()){
        printf("Check model info pass !!!\n");
        // auto result = worker->getOutputDirect();
        // if(result.empty()){
        //     printf("OUTPUT ERROR\n");
        // }
        // else{
        //     printf("result:\n");
        //     for(const auto & value: result){
        //         printf("%f ", value);
        //     }
        //     printf("\n");
        // }
    }
    else{
        printf("Check model info fail !!!\n");
    }

    delete worker;
    return 0;
}
