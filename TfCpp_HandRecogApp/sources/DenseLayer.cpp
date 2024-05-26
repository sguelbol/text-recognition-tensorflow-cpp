#include <tensorflow/cc/client/client_session.h>
#include "../headers/DenseLayer.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include <iostream>
#include <utility>

/**
 * Initializes the weights and biases tensors for the DenseLayer.
 *
 * This method initializes the weights and biases tensors of the DenseLayer object
 * with random values.
 *
 * @param session A shared pointer to*/
DenseLayer::DenseLayer(shared_ptr<ClientSession> session, const Scope& scope, int inputDim, int numberOfNeurons, ActivationFunction activationFunction) :
        session(move(session)),
        scope(scope),
        inputDim(inputDim),
        numberOfNeurons(numberOfNeurons),
        activationFunction(activationFunction) {
    this->initializeWeights();
}

/**
 * @brief Destructor for the DenseLayer class.
 *
 * This destructor is responsible for cleaning up any resources allocated by the DenseLayer object.
 *
 * @return None
 */
DenseLayer::~DenseLayer() = default;


/**
 * Initializes the weights and biases tensors for the DenseLayer.
 *
 * This method initializes the weights and biases tensors of the DenseLayer object
 * with random values.
 *
 * @param None
 * @return None
 */
void DenseLayer::initializeWeights() {
    // Initialize the weights tensor with random values
    auto randomWeights = RandomUniform(scope,Const(scope.WithOpName("output_shape"),{inputDim, numberOfNeurons}),DT_FLOAT);
    weights = make_shared<Variable>(Variable(scope, {inputDim, numberOfNeurons}, DT_FLOAT));
    auto initializeWeights = Assign(scope, *weights, randomWeights);

    auto randomBias = RandomUniform(scope, Const(scope.WithOpName("bias"), {1, numberOfNeurons}), DT_FLOAT);
    bias = make_shared<Variable>(Variable(scope, {1, numberOfNeurons}, DT_FLOAT));
    auto initializeBias = Assign(scope, *bias, randomBias);

    TF_CHECK_OK(session->Run({initializeWeights, initializeBias}, nullptr));
}


/**
 * Performs the initial forward pass of the first DenseLayer.
 *
 * This method takes an input image tensor and performs the initial forward pass
 * through the DenseLayer. It multiplies the input with the weights, adds the biases, and applies finally the activation function.
 * The final result is stored in the `output` member variable and will be returned.
 *
 * @note Nothing is calculated here, only the operations for the feed-forward of this layer are assembled.
 * The output tensor stores the operations for the forward pass, while the actual computation takes place in the model class .
 *
 * @see DenseLayer::subsequentForwardPass(Output previousLayerOutput)

 *
 * @param inputImage The input image tensor.
 * @return The assembled computations represented in an output tensor after the initial forward pass.
 */
Output DenseLayer::initialForwardPass(shared_ptr<Placeholder> inputImage) {
    if (weights == nullptr) { cout << "Nullptr" << endl; stderr;}
    auto matmul = MatMul(scope, *inputImage, *weights);
    auto addBias = Add(scope, matmul, *bias);
    this->output = this->applyActivationFunction(scope, addBias);
    return this->output;
}


/**
 * Performs the subsequent forward pass of the subsequent DenseLayer.
 *
 * This method takes the output of the previous layer as input and performs the
 * subsequent forward pass through the DenseLayer. It multiplies the output of the previous layer with the weights,
 * adds the biases and applies finally the activation function.
 * The final result is stored in the `output` member variable and will be returned.
 *
 * @note Nothing is calculated here, only the operations for the feed-forward of this layer are assembled.
 * The output of the previous layer is only the assembly of the operations of the previous layer.
 * The output tensor stores the operations for the forward pass, while the actual computation takes place in the model class .
 *
 * @see DenseLayer::initialForwardPass(shared_ptr<Placeholder> inputImage)
 *
 * @param previousLayerOutput The output tensor of the previous layer.
 * @return The output tensor of the DenseLayer after the subsequent forward pass.
 */
Output DenseLayer::subsequentForwardPass(Output previousLayerOutput) {
    if (weights == nullptr) { cout << "Nullptr" << endl;}
    auto matmul = MatMul(scope, previousLayerOutput, *weights);
    auto addBias = Add(scope, matmul, *bias);
    this->output = this->applyActivationFunction(scope, addBias);
    return this->output;
}

/**
 * Applies the specified activation function to the given input.
 *
 * This method takes the input tensor and applies the specified activation function
 * to it. The supported activation functions are sigmoid, relu, softmax, selu, and elu.
 *
 * @param scope The TF scope for the operation.
 * @param addBias The input tensor to be passed through the activation function.
 * @return The tensor resulting from the application of the activation function.
 */
Output DenseLayer::applyActivationFunction(Scope scope, Output addBias) {
    Output result;
    switch (activationFunction) {
        case ActivationFunction::SIGMOID:
            result = Sigmoid(scope.WithOpName("Activation"), addBias); break;
        case ActivationFunction::RELU:
            result = Relu(scope.WithOpName("Activation"), addBias); break;
        case ActivationFunction::SOFTMAX:
            result = Softmax(scope.WithOpName("Activation"), addBias); break;
        case ActivationFunction::SELU:
            result = Selu(scope.WithOpName("Activation"), addBias); break;
        case ActivationFunction::ELU:
            result = Elu(scope.WithOpName("Activation"), addBias); break;
    }
    return result;
}

/**
 * Prints the details of the DenseLayer.
 *
 * This method prints the unique name of the DenseLayer's scope, the input dimension, the output dimension,
 * the activation function, and the status of the scope.
 *
 * @param None
 * @return None
 */
void DenseLayer::printLayer() {
    cout << " --------- " << scope.GetUniqueNameForOp("") << " ---------" << endl;
    cout << "Input-Dimension: " << inputDim << " Number of neurons: " << numberOfNeurons << endl;
    cout << "Activation-Function: " << ActivationFunctionConverter::toString(activationFunction) << endl;
    cout << " (" << "Scope-Status: "<< (scope.ok()? "OK" : "Failed") << ")" << endl;
    //printWeights();
}

/**
 * Print the weights of the DenseLayer.
 *
 * This method prints the weights of the DenseLayer object to the console.
 *
 * @param None
 * @return None
 */
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

/**
 * Retrieves the dimension of input of the DenseLayer.
 *
 * This method returns the dimension of input of the DenseLayer.
 *
 * @return An integer value representing the dimension of the input in the DenseLayer.
 */
int DenseLayer::getInputDim() const {
    return inputDim;
}

/**
 * Retrieves the number of neurons of the DenseLayer.
 *
 * This method returns the number of neurons of neurons of the DenseLayer.
 *
 * @return An integer value representing the dimension of the neurons in the DenseLayer.
 */
int DenseLayer::getNumberOfNeurons() const {
    return numberOfNeurons;
}

/**
 * Returns the weights tensor of the DenseLayer.
 *
 * This method returns the `weights` tensor of the DenseLayer object.
 * Each layer has as much neurons as given by `numberOfNeurons`, the size of neurons in the layer
 * corresponds to the size of the output. Each neuron of the layer receives as much inputs as given by `inputDim`.
 * The input dim correspond in the case of the first layer to dimensionality of the input data.
 * In the case of a subsequent layer, it would be the output dimension of the previous layer.
 * So the weights tensor is shape [inputDim, numberOfNeurons].
 *
 * @param None
 * @return A shared pointer to the Variable class representing the weights tensor.
 */
const shared_ptr<Variable> &DenseLayer::getWeights() const {
    return weights;
}

/**
 * Returns the biases tensor of the DenseLayer.
 *
 * This method returns the `bias` tensor of the DenseLayer object.
 * The bias tensor contains the bias values for each neuron in the layer.
 *
 * @param None
 * @return A reference to the bias variable of the DenseLayer.
 */
const shared_ptr<Variable> &DenseLayer::getBiases() const {
    return bias;
}
