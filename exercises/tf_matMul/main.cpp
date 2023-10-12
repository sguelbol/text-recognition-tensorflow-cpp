#include <iostream>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/cc/client/client_session.h>

int main() {
    tensorflow::Scope root = tensorflow::Scope::NewRootScope();
    tensorflow::ClientSession session(root);

    std::vector<std::vector<float>> inputData = {{1, 0, 0},
                                                 {1, 0, 1},
                                                 {1, 1, 0},
                                                 {1, 1,1}};
    tensorflow::Tensor input(tensorflow::DT_FLOAT, tensorflow::TensorShape({4, 3}));
    auto inputMapped = input.tensor<float, 2>();
    for (int x = 0; x < 4; ++x) {
        for (int y = 0; y < 3; ++y) {
            inputMapped(x, y) = inputData[x][y];
        }
    }
    std::cout << "Inputs: " << input << std::endl;

    tensorflow::Tensor weights(tensorflow::DT_FLOAT, tensorflow::TensorShape({3, 1}));
    std::vector<std::vector<float>> weightData = {{1}, {1}, {1}};
    auto weightsMapped = weights.tensor<float, 2>();
    //std::copy(weightData.begin(), weightData.end(), weightsMapped.data());
    //std::cout << "Weights: " << weights << std::endl;
    for (int x = 0; x < 3; ++x) {
        weightsMapped(x, 0) = weightData[x][0];
    }

    auto m = tensorflow::ops::MatMul(root, input, weights);

    std::vector<tensorflow::Tensor> outputs;
    tensorflow::Status run_status = session.Run({}, {m}, {}, &outputs);
    if (!run_status.ok()) {
        std::cerr << "Error running session: " << run_status.ToString() << std::endl;
        // Handle the error
    }
    auto i = outputs[0].tensor<float, 2>();
    std::cout << i << std::endl;
    std::cout << outputs[0] << std::endl;

    //std::cout << "Result of addition: " <<     outputs[0].tensor<float, 2>() << std::endl;

    return 0;
}
