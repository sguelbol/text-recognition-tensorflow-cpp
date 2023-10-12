#include <iostream>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/cc/client/client_session.h>


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
    tensorflow::ClientSession session(root);

    std::vector<std::vector<float>> inputData = { {1, 0, 0},
                                                  {1, 0, 1},
                                                  {1, 1, 0},
                                                  {1, 1, 1} };
    initInput(inputData);

    std::vector<std::vector<float>> weightData = {{1}, {1}, {1}};
    initWeights(weightData);

    std::vector<std::vector<float>> expectedOutputData = {{0}, {0}, {0}, {1}};
    initExpectedOutput(expectedOutputData);

    initLearningRate(0.3);

    //tensorflow::Scope root2 = tensorflow::Scope::NewRootScope();
    tensorflow::ClientSession session2(root);

        auto matMul = tensorflow::ops::MatMul(root, input, weights);
        std::vector<tensorflow::Tensor> output;
        tensorflow::Status run_status = session2.Run({}, {matMul}, {}, &output);
        if (!run_status.ok()) {
            std::cerr << "Error running session: " << run_status.ToString() << std::endl;
            // Handle the error
        }
        auto tensor_y = threshold(output[0]);
        std::cout << "Result of y: " << tensor_y << std::endl;

        if (equal(tensor_y, expectedOutput)) {
            std::cout << "fi" << std::endl;
            break;} else {             std::cout << "not" << std::endl;
        }

        //Compute error
        auto sub = tensorflow::ops::Subtract(root, expectedOutput, tensor_y);
        std::vector<tensorflow::Tensor> w;
        session.Run({}, {sub}, {}, &w);
        auto tensor_error = w[0];
        std::cout << "Error: " << tensor_error << std::endl;

        //Compute delta
        tensorflow::ClientSession session3(root);
        auto delta_computation = tensorflow::ops::Multiply(root, tensorflow::ops::MatMul(root, input, tensor_error, tensorflow::ops::MatMul::TransposeA(true)), learningRate);
        std::vector<tensorflow::Tensor> delta;
        session3.Run({}, {delta_computation}, {}, &delta);
        auto tensor_product = delta[0];
        std::cout << "Delta: " << tensor_product << std::endl;

        //Update weights
        tensorflow::ClientSession session4(root);
        auto update_weights_computation = tensorflow::ops::Add(root, weights, tensor_product);
        std::vector<tensorflow::Tensor> updatedWeights;
        session4.Run({}, {update_weights_computation}, {}, &updatedWeights);
        auto updatedWeights2= updatedWeights[0];
        std::cout << "Updated_weights " << updatedWeights2 << std::endl;


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
    std::cout << "Threshold: " << calculated_y << std::endl;
    for (int i = 0; i < calculated_y.size(); ++i) {
        if (calculated_y(i) > 0) {
            calculated_y(i) = 1;
        } else {
            calculated_y(0);
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


