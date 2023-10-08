#include <iostream>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/cc/client/client_session.h>


void initInput(std::vector<std::vector<float>> inputData);
void initWeights(std::vector<float> weightData);
void initExpectedOutput(std::vector<float> expectedOutputData);
void initLearningRate(float learningRateData);

tensorflow::Tensor input(tensorflow::DT_FLOAT, tensorflow::TensorShape({4, 3}));
tensorflow::Tensor weights(tensorflow::DT_FLOAT, tensorflow::TensorShape({3}));
tensorflow::Tensor expectedOutput(tensorflow::DT_FLOAT, tensorflow::TensorShape({4}));
tensorflow::Tensor learningRate(tensorflow::DT_FLOAT, tensorflow::TensorShape({1}));

int main() {

    tensorflow::Scope root = tensorflow::Scope::NewRootScope();
    tensorflow::ClientSession session(root);

    std::vector<std::vector<float>> inputData = {{1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};
    initInput(inputData);

    std::vector<float> weightData = {1, 1, 1};
    initWeights(weightData);

    std::vector<float> expectedOutputData = {0, 0, 0, 1};
    initExpectedOutput(expectedOutputData);

    initLearningRate(0.3);

    auto m = tensorflow::ops::MatMul(root, input, weights);
    //std::cout << m << std::endl;


kj
    std::vector<tensorflow::Tensor> outputs;
    //session.Run();


    return 0;
}


void initInput(std::vector<std::vector<float>> inputData) {
    auto inputMapped = input.tensor<float, 2>();
    for (int x = 0; x < 4; ++x) {
        for (int y = 0; y < 3; ++y) {
            inputMapped(x, y) = inputData[x][y];
        }
    }
    std::cout << "Inputs: " << input << std::endl;
}

void initWeights(std::vector<float> weightData) {
    auto weightsMapped = weights.vec<float>();
    std::copy(weightData.begin(), weightData.end(), weightsMapped.data());
    std::cout << "Weights: " << weights << std::endl;
}

void initExpectedOutput(std::vector<float> expectedOutputData) {
    auto expectedOutputsMapped = expectedOutput.vec<float>();
    std::copy(expectedOutputData.begin(), expectedOutputData.end(), expectedOutputsMapped.data());
    std::cout << "Expected outputs: " << expectedOutput << std::endl;
}

void initLearningRate(float learningRateData) {
    learningRate.scalar<float>()() = learningRateData;
    std::cout << "Learning rate: " << learningRate << std::endl;
}


