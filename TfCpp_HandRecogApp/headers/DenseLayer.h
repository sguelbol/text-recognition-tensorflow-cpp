#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/ops/state_ops.h>
#include <tensorflow/cc/framework/ops.h>
#include "../enum/ActivationFunction.h"

#ifndef MULTILAYERPERCEPTRON_LAYER_H
#define MULTILAYERPERCEPTRON_LAYER_H

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

/**
 * @class DenseLayer
 * @brief Represents a dense layer in a neural network.
 *
 * The DenseLayer class is designed to create a dense layer in a neural network.
 */
class DenseLayer {
public:
    DenseLayer(shared_ptr<ClientSession> session, const Scope& scope, int inputDim, int numberOfNeurons, ActivationFunction activation_Function);
    ~DenseLayer();

    //Methods
    Output initialForwardPass(shared_ptr<Placeholder> inputImage);
    Output subsequentForwardPass(Output previousLayerOutput);
    void printLayer();
    void printWeights();
    int getNumberOfNeurons() const;
    const shared_ptr<Variable> &getWeights() const;
    const shared_ptr<Variable> &getBiases() const;


private:
    // Variables
    shared_ptr<ClientSession> session;
    Scope scope = Scope::NewRootScope();
    int inputDim;
    int numberOfNeurons;
    shared_ptr<Variable> weights;
    shared_ptr<Variable> bias;
    Output output;
    ActivationFunction activationFunction;
    //Methods
    void initializeWeights();
    Output applyActivationFunction(Scope scope, Output addBias);
};

#endif //MULTILAYERPERCEPTRON_LAYER_H