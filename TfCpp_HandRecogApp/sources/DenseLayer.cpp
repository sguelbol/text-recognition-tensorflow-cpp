#include <tensorflow/cc/client/client_session.h>
#include "../headers/DenseLayer.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include <iostream>
#include <utility>

DenseLayer::DenseLayer(shared_ptr<ClientSession> session, const Scope& scope, int inputDim, int outputDim, ActivationFunction activationFunction) :
        session(move(session)),
        scope(scope),
        inputDim(inputDim),
        outputDim(outputDim),
        activationFunction(activationFunction) {
    this->initializeWeights();
}

DenseLayer::~DenseLayer() = default;


void DenseLayer::initializeWeights() {
    // Initialize the weights tensor with random values
    auto randomWeights = RandomUniform(scope,Const(scope.WithOpName("output_shape"),{inputDim, outputDim}),DT_FLOAT);
    weights = make_shared<Variable>(Variable(scope, {inputDim, outputDim}, DT_FLOAT));
    auto initializeWeights = Assign(scope, *weights, randomWeights);

    auto randomBias = RandomUniform(scope, Const(scope.WithOpName("bias"), {1, outputDim}), DT_FLOAT);
    bias = make_shared<Variable>(Variable(scope, {1, outputDim}, DT_FLOAT));
    auto initializeBias = Assign(scope, *bias, randomBias);

    TF_CHECK_OK(session->Run({initializeWeights, initializeBias}, nullptr));
}

Output DenseLayer::forward(shared_ptr<Placeholder> input) {
    if (weights == nullptr) { cout << "Nullptr" << endl; stderr;}
    auto matmul = MatMul(scope, *input, *weights);
    auto addBias = Add(scope, matmul, *bias);

    switch (activationFunction) {
        case ActivationFunction::SIGMOID:
            this->output = Sigmoid(scope.WithOpName("Activation"), addBias); break;
        case ActivationFunction::RELU:
            this->output = Relu(scope.WithOpName("Activation"), addBias); break;
        case ActivationFunction::SOFTMAX:
            this->output = Softmax(scope.WithOpName("Activation"), addBias); break;
    }
    return this->output;
}

Output DenseLayer::forward(Output output) {
    if (weights == nullptr) { cout << "Nullptr" << endl;}
    auto matmul = MatMul(scope, output, *weights);
    auto addBias = Add(scope, matmul, *bias);
    this->output = Sigmoid(scope.WithOpName("Activation"), addBias);
    return this->output;
}

void DenseLayer::printLayer() {
    cout << " --------- " << scope.GetUniqueNameForOp("") << " ---------" << endl;
    cout << "InputDim: " << inputDim << " OutputDim: " << outputDim << endl;
    cout << "Activation-Function: " << ActivationnFunctionConverter::toString(activationFunction) << endl;
    cout << " (" << "Scope-Status: "<< (scope.ok()? "OK" : "Failed") << ")" << endl;
    //printWeights();
}

void DenseLayer::printWeights() {
    vector<Tensor> output;
    TF_CHECK_OK(session->Run({*weights}, &output));
    cout << "Weights: " << endl;
    for (int i = 0; i < output[0].dim_size(0); i++) {
        cout << "\t";
        cout << "[ ";
        for (int j = 0; j < output[0].dim_size(1); j++) {
            cout << output[0].matrix<float>()(i, j) << " ";
        }
        cout << "]" << endl;
    }
}

int DenseLayer::getOutputDim() const {
    return outputDim;
}

const shared_ptr<Variable> &DenseLayer::getWeights() const {
    return weights;
}

const shared_ptr<Variable> &DenseLayer::getBiases() const {
    return bias;
}
