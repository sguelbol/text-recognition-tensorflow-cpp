#include <iostream>
#include <core/framework/tensor.h>

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/core/framework/stats_aggregator.h>



void initInput(std::vector<std::vector<float>> inputData);

void initWeights(std::vector<std::vector<float>> weightData);

void initExpectedOutput(std::vector<std::vector<float>> expectedOutputData);

void initLearningRate(float learningRateData);

tensorflow::Tensor threshold(tensorflow::Tensor y);

bool equal(tensorflow::Tensor a, tensorflow::Tensor b);


tensorflow::Tensor input(tensorflow::DT_FLOAT, tensorflow::TensorShape({4, 3}));
tensorflow::Tensor weights(tensorflow::DT_FLOAT, tensorflow::TensorShape({3, 1}));
tensorflow::Tensor expectedOutput(tensorflow::DT_FLOAT, tensorflow::TensorShape({4, 1}));
tensorflow::Tensor learningRate(tensorflow::DT_FLOAT, tensorflow::TensorShape({1}));


int main() {
    tensorflow::Scope root = tensorflow::Scope::NewRootScope();

    std::vector<std::vector<float>> inputData = {{1, 0, 0},
                                                 {1, 0, 1},
                                                 {1, 1, 0},
                                                 {1, 1, 1}};
    initInput(inputData);

    std::vector<std::vector<float>> weightData = {{1},
                                                  {1},
                                                  {1}};
    initWeights(weightData);

    auto tg = tensorflow::Input::Initializer({1, 2, 3});
    //https://stackoverflow.com/questions/39148671/how-to-fill-a-tensor-in-c
    std::cout << "initializer" << tg.tensor << std::endl;

    //and-function
    std::vector<std::vector<float>> expectedOutputData = {{0},
                                                          {0},
                                                          {0},
                                                          {1}};
    initExpectedOutput(expectedOutputData);

    initLearningRate(0.3);

    auto weights_placeholder = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);

    tensorflow::ClientSession session1(root);

    /*auto matMul = tensorflow::ops::MatMul(root, input_placeholder, weights);
    auto sub = tensorflow::ops::Subtract(root, expectedOutput, matMul);
    std::vector<tensorflow::Tensor> w;
    auto input2 = tensorflow::Input({{1.0f, 0.0f, 0.0f},
                                     {1.0f, 0.0f, 1.0f},
                                     {1.0f, 1.0f, 0.0f},
                                     {1.0f, 1.0f, 1.0f}});
    TF_CHECK_OK(session1.Run({ {input_placeholder, input}} , {sub}, &w));
    auto tensor_error = w[0];
    std::cout << "Error: " << tensor_error << std::endl;
    //input_placeholder.operator ::tensorflow::Input(;


    auto a = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
    auto c = tensorflow::ops::Add(root, a, {41});
    std::vector<tensorflow::Tensor> outputs;
    session1.Run({ {a, {1} } }, {c}, &outputs);
    std::cout << "R: " << outputs[0] << std::endl; */

    while (true) {
        std::cout << "Weights: " << weights << std::endl;

        std::vector<tensorflow::Tensor> output;
        auto matMul = tensorflow::ops::MatMul(root, input, weights_placeholder);
        tensorflow::ClientSession session1(root);

        TF_CHECK_OK(session1.Run({{weights_placeholder, weights}}, {matMul}, &output));
        auto tensor_y = threshold(output[0]);
        std::cout << "Result of y: " << tensor_y << std::endl;

        if (equal(tensor_y, expectedOutput)) {
            std::cout << "fi" << std::endl;
            break;
        } else {
            std::cout << "not" << std::endl;
        }

        //Compute error
        auto sub = tensorflow::ops::Subtract(root, expectedOutput, tensor_y);

        //Compute delta
        auto delta_computation = tensorflow::ops::Multiply(root, tensorflow::ops::MatMul(root, input, sub, tensorflow::ops::MatMul::TransposeA(true)), learningRate);

        //Update weights
        tensorflow::ClientSession session2(root);
        auto update_weights_computation = tensorflow::ops::Add(root, weights, delta_computation);
        std::vector<tensorflow::Tensor> updatedWeights;
        TF_CHECK_OK(session2.Run({}, {update_weights_computation}, {}, &updatedWeights));
        auto updatedWeights2 = updatedWeights[0];
        std::cout << "Updated_weights " << updatedWeights2 << std::endl;
        weights = updatedWeights2;
    }
}

bool equal(tensorflow::Tensor a, tensorflow::Tensor b) {
    auto a_mapped = a.flat<float>();
    auto b_mapped = b.flat<float>();
    if (a_mapped.size() != b_mapped.size()) {
        return false;
    }
    for (int y = 0; y < a_mapped.size(); ++y) {
        if (a_mapped(y) != b_mapped(y)) return false;
    }
    return true;
}

tensorflow::Tensor threshold(tensorflow::Tensor y) {
    auto calculated_y = y.flat<float>();
    std::cout << "Before Threshold: " << calculated_y << std::endl;
    for (int i = 0; i < calculated_y.size(); ++i) {
        if (calculated_y(i) > 0) {
            calculated_y(i) = 1;
        } else {
            calculated_y(i) = 0;
        }
    }
    std::cout << "Threshold: " << calculated_y << std::endl;
    return y;
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

void initWeights(std::vector<std::vector<float>> weightData) {
    auto weightsMapped = weights.tensor<float, 2>();
    for (int x = 0; x < 3; x++) {
        weightsMapped(x, 0) = weightData[x][0];
    }
    std::cout << "Weights: " << weights << std::endl;
}

void initExpectedOutput(std::vector<std::vector<float>> expectedOutputData) {
    auto expectedOutputsMapped = expectedOutput.tensor<float, 2>();
    for (int x = 0; x < 4; x++) {
        expectedOutputsMapped(x, 0) = expectedOutputData[x][0];
    }
    std::cout << "Expected outputs: " << expectedOutput << std::endl;
}

void initLearningRate(float learningRateData) {
    learningRate.scalar<float>()() = learningRateData;
    std::cout << "Learning rate: " << learningRate << std::endl;
}


