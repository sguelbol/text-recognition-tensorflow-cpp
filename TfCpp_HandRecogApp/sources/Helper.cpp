#include "../headers/Helper.h"
#include <tensorflow/cc/client/client_session.h>

void Helper::printImageInConsole(Tensor &image) {
    std::cout << image.shape() << std::endl;
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; ++j) {
            std::cout << image.flat<float>()(28*i+j) << " ";
        }
        std::cout << std::endl;
    }
}

void Helper::printLabelInConsole(Tensor &label) {
    std::cout << "Label: " << std::endl;
    std::cout << label.flat<float>() << std::endl;
}

Tensor Helper::calculatePredictedClass(Tensor &modelOutput) {
    Scope scope = Scope::NewRootScope();
    ClientSession session(scope);
    auto predictedClass = ArgMax(scope, modelOutput, 1);
    vector<Tensor> output;
    session.Run({predictedClass}, &output);
    return output[0];
}


