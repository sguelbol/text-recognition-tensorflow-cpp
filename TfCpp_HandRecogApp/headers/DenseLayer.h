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

class DenseLayer {
public:
    DenseLayer(shared_ptr<ClientSession> session, const Scope& scope, int inputDim, int outputDim, ActivationFunction activation_Function);
    ~DenseLayer();

    //Methods
    Output forward(shared_ptr<Placeholder> input);
    Output forward(Output output);
    void printLayer();
    void printWeights();
    int getOutputDim() const;
    const shared_ptr<Variable> &getWeights() const;
    const shared_ptr<Variable> &getBiases() const;


private:
    // Variables
    shared_ptr<ClientSession> session;
    Scope scope = Scope::NewRootScope();
    int inputDim;
    int outputDim;
    shared_ptr<Variable> weights;
    shared_ptr<Variable> bias;
    Output output;
    ActivationFunction activationFunction;
    //Methods
    void initializeWeights();
};

#endif //MULTILAYERPERCEPTRON_LAYER_H