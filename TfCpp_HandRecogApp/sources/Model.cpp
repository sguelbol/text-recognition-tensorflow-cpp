#include "../headers/Model.h"

#include <memory>
#include "../headers/DenseLayer.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/summary/summary_file_writer.h"
#include "../headers/Helper.h"
#include "tensorflow/cc/framework/gradients.h"



Model::Model(Scope &modelScope) : scope(modelScope), session(std::make_shared<ClientSession>(scope)) {
}

Model::~Model() = default;


/**
 * @brief Adds an input layer to the model.
 *
 * This function creates an input layer in the model with the specified input dimension which is the initial layer of the model where the raw input data are fed.
 * The `featuresDim` member variable is initialized with a dynamic memory allocation using `std::make_unique<int>()` to store the input dimension.
 * The `features` member variable is initialized with a shared pointer to a `Placeholder` object, which represents the input features tensor.
 * The placeholder is created using the `inputScope` sub-scope and named "Feature". The data type is set to `DT_FLOAT` and the shape is specified as `{-1, (*featuresDim)}`.
 *
 * @param inputDim The dimension of the input features tensor.
 */
void Model::addInputLayer(int inputDim) {
    Scope inputScope = scope.NewSubScope("Input-Layer");
    featuresDim = std::make_unique<int>(inputDim);
    features = std::make_shared<Placeholder>(inputScope.WithOpName("Feature"), DT_FLOAT, Placeholder::Shape({-1, *featuresDim}));
    labels = std::make_shared<Placeholder>(inputScope.WithOpName("Label"), DT_FLOAT, Placeholder::Shape({-1, *labelsDim}));
}

/**
 * @brief Adds a dense layer to the model.
 *
 * This function creates and adds a dense layer to the model. The dense layer is initialized with the specified input
 * dimension and output dimension. The input dimension is the output dimension of the previous layer, whereas the
 * first dense-layer has no previous dense-layer it takes the input dimension from the input-layer.
 *
 * @param outputDim The number of neurons in the dense layer.
 */
void Model::addDenseLayer(int neuronsPerLayer, ActivationFunction activationFunction) {
    Scope denseLayerScope = scope.NewSubScope("Dense-Layer-" + std::to_string(layers.size()+1));
    std::unique_ptr<DenseLayer> denseLayer;
    int inputDim = *featuresDim;
    if (!this->layers.empty()) {
        auto &previousLayer = layers[layers.size() - 1];
        inputDim = previousLayer->getOutputDim();
    }
    *labelsDim = neuronsPerLayer;
    denseLayer = std::make_unique<DenseLayer>(this->session, denseLayerScope, inputDim, neuronsPerLayer, activationFunction);
    layers.push_back(std::move(denseLayer));
}


void Model::buildModel() {
    if (layers.empty()) {
        std::cerr << "No dense layers, at least one dense layer should be added to model!";
        std::exit(EXIT_FAILURE);
    }
    bool isFirstLayer = true;
    for (auto& layer : layers) {
        if (isFirstLayer) {
            this->model = layer->forward(this->features);
            isFirstLayer = false;
        } else {
            this->model = layer->forward(this->model);
        }
    }
}

Tensor Model::predict(Tensor inputFeatures) {
    std::vector<Tensor> outputs;
    inputFeatures = reshapeInput(inputFeatures);
    TF_CHECK_OK(session->Run({{*features, inputFeatures}}, {model}, {}, &outputs));
    return outputs[0];
}


Tensor Model::reshapeInput(Tensor inputFeatures) {
    if (inputFeatures.dims() == 1) {
        vector<Tensor> outputs;
        auto reshaped = Reshape(scope, inputFeatures, {1, *featuresDim});
        session->Run({reshaped}, &outputs);
        return outputs[0];
    }
    return inputFeatures;
}

void Model::retrain(Tensor imageTensor, int expectedNumber) {
    Scope retrainScope = scope.NewSubScope("Retrain");
    auto onehot = OneHot(retrainScope, {expectedNumber}, Input::Initializer(10), Input::Initializer(1.0f), Input::Initializer(0.0f));
    vector<Tensor> output;
    session->Run({onehot}, &output);
    this->train(imageTensor, output[0], 10, 0.7f, 1);
}

/**
 * @brief Prints the details of the model.
 *
 * This function outputs the details of the model to the command-line. It prints each layer, followed by the details of each layer in the model.
 * For each layer, the `printLayer()` method is called to print the details of that layer.
 *
 * The `#` characters are used to visually separate the model details.
 *
 * @see DenseLayer::printLayer()
 *
 * @note This method does not return any data, but only prints the details of the model to the standard output stream.
 */
void Model::printModel() {
    std::cout << "#-#-#-#-#-#-#-#-#-#-#-#-#-# " << " Model-Details " << " #-#-#-#-#-#-#-#-#-#-#-#-#-#" << std::endl;
    for (const auto &layer : layers) {
        std::cout << "" << std::endl;
        layer->printLayer();
    }
    std::cout << "" << std::endl;
    std::cout << "#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#" << std::endl;
}


void Model::printOutput(const Tensor& output) {
    for (int i = 0; i < output.dim_size(0); i++) {
        for (int j = 0; j < output.dim_size(1); j++) {
            std::cout << output.matrix<float>()(i, j) << " ";
        }
        std::cout << "\n";
    }
    //std::cout << output.matrix<float>() << std::endl;
}

void Model::train(Tensor imageTensor, Tensor labelTensor, int maxEpochs, float learningRate, int batchSize) {
    std::cout << "#-#-#-#-#-#-#-#-#-#-#-#-#-#-# " << " Training " << " #-#-#-#-#-#-#-#-#-#-#-#-#-#-#" << std::endl;
    if (imageTensor.dim_size(0) != labelTensor.dim_size(0)) {
        std::cerr << "Image und label dataset size must fit together";
        std::exit(EXIT_FAILURE);
    }
    Tensor imageBatches, labelBatches;
    std::tie(imageBatches, labelBatches) = getBatches(batchSize, imageTensor, labelTensor);
    Scope lossScope = scope.NewSubScope("Training");
    auto loss = Mean(lossScope.WithOpName("Loss"), SquaredDifference(lossScope.WithOpName("Sigmoid-Cross-Entropy"), model, *this->labels), {0});
    std::vector<Output> apply_gradients = this->backpropagation(lossScope,learningRate, loss);

    int dataSize = imageBatches.dim_size(0);
    std::vector<Tensor> outputs;
    Tensor imageBatch;
    Tensor labelBatch;
    for (int i = 1; i <= maxEpochs; i++) {
        auto lossValue = 0;
        vector<Tensor> output1;
        for (int64_t num = 0; num < dataSize; num++) {
            //auto d1 = DeepCopy(scope, imageBatches.SubSlice(num));
            //auto d2 = DeepCopy(scope, labelBatches.SubSlice(num));
            //TF_CHECK_OK(session->Run({d1, d2}, &output1));
            imageBatch = imageBatches.SubSlice(num);
            labelBatch = labelBatches.SubSlice(num);
            //TODO: Only Batches of num % 8 = 0 allowed because of alignment error

            TF_CHECK_OK(session->Run({{*features, imageBatch}, {*this->labels, labelBatch}}, apply_gradients, {}, nullptr));
            if (num == dataSize) {
                //TF_CHECK_OK(session->Run({{*features, inputFeatures[num]}, {*this->labels, labels[num]}}, {loss}, &outputs));
            }
        }
        if (i % 10 == 0) {
            TF_CHECK_OK(session->Run({{*features, imageBatch}, {*this->labels, labelBatch}}, {loss}, &outputs));
            std::cout << " " << std::endl;
            std::cout << "Epoch " << i << " Loss: " << outputs[0].flat<float>() << std::endl;
        }
    }
    std::cout << " " << std::endl;
    std::cout << "#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#" << std::endl;
}

std::vector<Output> Model::backpropagation(Scope lossScope, float learningRate, Output loss) {
    vector<shared_ptr<Variable>> weights = getAllLayerWeights();
    vector<shared_ptr<Variable>> biases = getAllLayerBiases();
    // Combine weights and biases into one vector
    std::vector<Output> weights_and_biases;
    for (shared_ptr<Variable> w : weights) {
        weights_and_biases.push_back(*w);
    }
    for (shared_ptr<Variable> b : biases) {
        weights_and_biases.push_back(*b);
    }
    vector<Output> gradients;
    TF_CHECK_OK(AddSymbolicGradients(lossScope.WithOpName("Gradients"), {loss}, weights_and_biases, &gradients));

    std::vector<Output> apply_gradients;
    for (int i = 0; i < weights.size(); i++) {
        Output apply_gradient = ApplyGradientDescent(lossScope.WithOpName("Apply-Gradients-W" + std::to_string(i)), *weights[i], Cast(scope, learningRate,  DT_FLOAT), gradients[i]);
        apply_gradients.push_back(apply_gradient);
    }
    for (int i = 0; i < biases.size(); i++) {
        Output apply_gradient = ApplyGradientDescent(lossScope.WithOpName("Apply-Gradients-B" + std::to_string(i)), *biases[i], Cast(scope, learningRate,  DT_FLOAT), gradients[i + weights.size()]);
        apply_gradients.push_back(apply_gradient);
    }
    return apply_gradients;
}

std::vector<std::shared_ptr<Variable>> Model::getAllLayerWeights() {
    std::vector<std::shared_ptr<Variable>> allWeights;
    for (auto& layer : layers) {
        allWeights.push_back(layer->getWeights());
    }
    return allWeights;
}

std::vector<std::shared_ptr<Variable>> Model::getAllLayerBiases() {
    std::vector<std::shared_ptr<Variable>> allBiases;
    for (auto& layer : layers) {
        allBiases.push_back(layer->getBiases());
    }
    return allBiases;
}


tuple<Tensor, Tensor> Model::getBatches(int batchSize, const Tensor &images, const Tensor &labels) {
    Tensor batchImages, batchLabels;
    Scope batchesScope = scope.NewSubScope("BatchesGenerator");
    auto dataSetSize = images.dim_size(0);
    int numBatches = dataSetSize / batchSize;
    //calculates how many images will be used for the batches, a few images will be ignored which won't fit into a batch
    int dataForBatches = numBatches * batchSize;
    Placeholder data(batchesScope, DT_FLOAT, Placeholder::Shape({dataSetSize, -1}));
    auto sliced = Slice(batchesScope, data, {0, 0}, {dataForBatches, -1});
    auto flatten = Reshape(batchesScope, sliced, {-1});
    auto reshapeToBatches = Reshape(batchesScope, flatten, {numBatches, batchSize, -1});
    vector<Tensor> outputs;
    TF_CHECK_OK(session->Run({{data, images}}, {reshapeToBatches}, &outputs));
    batchImages = outputs[0];
    TF_CHECK_OK(session->Run({{data, labels}}, {reshapeToBatches}, &outputs));
    batchLabels = outputs[0];
    return std::make_tuple(batchImages, batchLabels);
}

void Model::validate(Tensor imageTensor, Tensor labelTensor) {
    std::cout << "#-#-#-#-#-#-#-#-#-#-#-#-#-#-# " << " Validation " << " #-#-#-#-#-#-#-#-#-#-#-#-#-#-#" << std::endl;
    std::cout << "" << std::endl;
    int numCorrect = 0;
    std::vector<Tensor> outputs;
    Tensor predictedValues = predict(imageTensor);
    Tensor predictions = Helper::calculatePredictedClass(predictedValues);
    Tensor labels = Helper::calculatePredictedClass(labelTensor);
    for (int i = 0; i < predictedValues.dim_size(0); i++) {
        if (predictions.flat<int64>()(i) == labels.flat<int64>()(i)) {
            numCorrect++;
        }
    }
    auto accuracyRate= static_cast<float>(numCorrect)/predictedValues.dim_size(0);
    std::cout << "Accuracy Rate: "<< numCorrect << "/" << predictedValues.dim_size(0) << ", " << accuracyRate << std::endl;

    std::cout << "" << std::endl;
    std::cout << "#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#" << std::endl;
}
