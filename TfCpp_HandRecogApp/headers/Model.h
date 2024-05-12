#include "DenseLayer.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include <tensorflow/cc/client/client_session.h>
#include "../enum/ActivationFunction.h"
#ifndef MULTILAYERPERCEPTRON_MODEL_H
#define MULTILAYERPERCEPTRON_MODEL_H

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

/**
 * @class Model
 *
 * @brief The Model class represents a neural network model.
 *        It allows building and training the model, predicting on input data
 *        and training on a single written character.
 *
 * @note The model assumes that the input features are flattened images represented as tensors.
 */
class Model {
public:
    Model(Scope &modelScope);
    ~Model();

    // Methods
    void addInputLayer(int inputDim);
    void addDenseLayer(int numberOfNeurons, ActivationFunction activationFunction);
    void buildModel();
    void printModel();
    Tensor predict(Tensor inputFeatures);

    void train(Tensor imageTensor, Tensor labelTensor, int maxEpochs, float learningRate, int batchSize);
    tuple<Tensor, Tensor> getBatches(int batchSize, const Tensor &images, const Tensor &labels);
    void validate(Tensor imageTensor, Tensor labelTensor);
    void trainOnWrittenChar(Tensor imageTensor, int expectedNumber);

private:
    // Variables
    Scope scope;
    shared_ptr<ClientSession> session;
    unique_ptr<int> featuresDim;
    unique_ptr<int> labelsDim = make_unique<int>(1);
    shared_ptr<Placeholder> features;
    shared_ptr<Placeholder> labels;
    vector<unique_ptr<DenseLayer>> layers;
    Output model;

    // Methods
    vector<shared_ptr<Variable>> getAllLayerWeights();
    vector<shared_ptr<Variable>> getAllLayerBiases();
    vector<Output> backpropagation(Scope lossScope, float learningRate, Output loss);
    Tensor reshapeInput(Tensor inputFeatures);
};

#endif //MULTILAYERPERCEPTRON_MODEL_H