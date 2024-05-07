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


/**
 * @brief Builds the model.
 *
 * This method builds the model by getting the operations of the forward pass of each dense layer in the model.
 * If it is the first layer, the `forward` method is called with `this->features` as the input. Otherwise, it is called with `this->model`
 * which is the calculations of the layers before.
 *
 * @see DenseLayer::forward(shared_ptr<Placeholder> input)
 * @see DenseLayer::forward(Output output)
 *
 * @note At least one dense layer should be added to the model before calling this method.
 */
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


/**
 * @brief Generates predictions using the model.
 *
 * This method takes in a `Tensor` object representing the input features and returns a `Tensor` object representing the model's predictions.
 *
 * @param inputFeatures The input features as a `Tensor` object.
 * @return The model's predictions as a `Tensor` object.
 */
Tensor Model::predict(Tensor inputFeatures) {
    std::vector<Tensor> outputs;
    inputFeatures = reshapeInput(inputFeatures);
    TF_CHECK_OK(session->Run({{*features, inputFeatures}}, {model}, {}, &outputs));
    return outputs[0];
}


/**
 * @brief Reshapes the input features tensor if it has only one dimension.
 *
 * This method checks if the input features tensor has only one dimension. If it does, it reshapes the tensor to have a shape of {1, *featuresDim},
 * this is necessary because the model expects input as tenosr of shape {n, *featuresDim}, n = 1 for single input, n > 1 for batch of inputs.
 * If the input features tensor has more than one dimension, it is returned as is.
 *
 * @param inputFeatures The input features tensor.
 * @return The reshaped input features tensor or the original tensor if it has more than one dimension.
 */
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


/**
 * @brief Prints the output tensor.
 *
 * This method prints the values in the output tensor to the console. It iterates over each element in the tensor using a nested loop and prints each value followed by a space.
 * After each row, it prints a newline character to move to the next row.
 *
 * @param output The output tensor to be printed.
 */
void Model::printOutput(const Tensor& output) {
    for (int i = 0; i < output.dim_size(0); i++) {
        for (int j = 0; j < output.dim_size(1); j++) {
            std::cout << output.matrix<float>()(i, j) << " ";
        }
        std::cout << "\n";
    }
    //std::cout << output.matrix<float>() << std::endl;
}

/**
 * @brief Trains the model on the provided image and label tensors.
 *
 * This function trains the model using the provided image and label tensors for a given number of epochs.
 * The image and label tensors are passed as parameters `imageTensor` and `labelTensor`, respectively.
 * The maximum number of epochs is specified by the parameter `maxEpochs`.
 * The learning rate for the training is specified by the parameter `learningRate`.
 * The batch size used for training is specified by the parameter `batchSize`.
 *
 * The image and label tensors are divided into batches using the `getBatches` function, and the resulting
 * batches are stored in the variables `imageBatches` and `labelBatches`.
 *
 * A new sub-scope named "Training" is created under the main scope, and a loss tensor is calculated
 * using the `Mean` and `SquaredDifference` functions. The loss tensor is named "Loss" and its operation
 * name is "Sigmoid-Cross-Entropy".
 *
 * The backpropagation step is performed using the `backpropagation` function, which returns a vector
 * of `Output` objects. These outputs are stored in the `apply_gradients` variable.
 *
 * The training step is performed by calling the `session->Run` function with the appropriate inputs,
 * gradients, and outputs. The inputs are the image and label tensors, the gradients are the outputs
 * of the `backpropagation` function, and the outputs are empty in this case. This step updates the model
 * parameters based on the gradients calculated during backpropagation.
 *
 * After each `batchSize` number of iterations, the loss is calculated by calling `session->Run` with the
 * current image and label batch tensors. The loss value is stored in the `outputs` vector, and if the current epoch
 * is a multiple of 10, the loss value is printed to the standard output stream.
 *
 * Finally, after all epochs are completed, a message is printed to the standard output stream indicating
 * the end of training.
 *
 * This function does not return any value.
 *
 * @param imageTensor The tensor containing the images for training.
 * @param labelTensor The tensor containing the labels for training.
 * @param maxEpochs The maximum number of epochs to train the model.
 * @param learningRate The learning rate for the training.
 * @param batchSize The batch size to use for training.
 */
void Model::train(Tensor imageTensor, Tensor labelTensor, int maxEpochs, float learningRate, int batchSize) {
    auto start = std::chrono::high_resolution_clock::now();

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

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Training Time elapsed: " << elapsed.count() << "ms" << std::endl;

}

/**
 * @brief Performs backpropagation to update the weights and biases of the model.
 *
 * This function computes the gradients of the loss with respect to the weights and biases of all layers in the model, and then applies the gradients to update the weights and biases using the ApplyGradientDescent operation.
 *
 * @param lossScope The scope for the loss subgraph.
 * @param learningRate The learning rate used for the gradient descent optimization.
 * @param loss The loss output tensor of the model.
 * @return A vector of output tensors representing the apply gradients operations for the weights and biases.
 */
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

/**
 * @brief Retrieves the weights of all layers in the model.
 *
 * This method iterates over each layer in the model and retrieves the weights of each layer.
 *
 * @return `std::vector` of `std::shared_ptr<Variable>` containing the weights of all layers in the model.
 */
std::vector<std::shared_ptr<Variable>> Model::getAllLayerWeights() {
    std::vector<std::shared_ptr<Variable>> allWeights;
    for (auto& layer : layers) {
        allWeights.push_back(layer->getWeights());
    }
    return allWeights;
}

/**
 * @brief Retrieves the biases of all layers in the model.
 *
 * This method iterates over each layer in the model and retrieves the biases of each layer.
 *
 * @return A `std::vector` of `std::shared_ptr<Variable>` containing the biases of all layers in the model.
 */
std::vector<std::shared_ptr<Variable>> Model::getAllLayerBiases() {
    std::vector<std::shared_ptr<Variable>> allBiases;
    for (auto& layer : layers) {
        allBiases.push_back(layer->getBiases());
    }
    return allBiases;
}


/**
 * @brief Generates batches of input images and labels.
 *
 * This function takes in a batch size, and two tensor objects `images` and `labels` representing the full dataset of images and labels.
 *
 * The function calculates the data set size by retrieving the dimension size of the `images` tensor object.
 * The number of batches, `numBatches`, is determined by dividing the data set size by the batch size.
 * The variable `dataForBatches` is calculated as the product of `numBatches` and `batchSize`.
 *
 * The `Slice` operation is used to slice the `data` placeholder object based on the range `{0, 0}` to `{dataForBatches, -1}`.
 * The resulting sliced tensor is stored in the `sliced` variable.
 *
 * The `Reshape` operation is used to flatten the `sliced` tensor into a 2D tensor with shape `{-1}`.
 * The resulting flattened tensor is stored in the `flatten` variable.
 *
 * Another `Reshape` operation is used to reshape the `flatten` tensor into a 3D tensor with shape `{numBatches, batchSize, -1}`.
 * The resulting reshaped tensor representing the batches of images is stored in the `reshapeToBatches` variable.
 *
 * The `session` member variable is used to run the TensorFlow session with two feed dicts:
 * {{data, images}} and {{data, labels}}.
 * The output tensors are stored in the `outputs` vector.
 *
 * Finally, the batch images and batch labels are extracted from the `outputs` vector and returned as a tuple.
 *
 * @param batchSize The size of each batch.
 * @param images The input tensor object representing the full dataset of images.
 * @param labels The input tensor object representing the full dataset of labels.
 * @return A tuple containing two Tensor objects: `batchImages` representing the batch of images, and `batchLabels` representing the batch of labels.
 */
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

/**
 * @brief Performs validation on the input image tensor and label tensor for one-hot labels.
 *
 * This method calculates the predicted values for the input image tensor using the `predict` method.
 * The class of predicted and label tensor is calculated using the `calculatePredictedClass` method from the `Helper` class.
 * The number of correct predictions is calculated by comparing the predicted class with the label class for each element of the predicted values.
 *
 * @see Helper::calculatePredictedClass
 *
 * @param imageTensor The tensor containing the input images.
 * @param labelTensor The tensor containing the corresponding labels for the input images.
 */
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
