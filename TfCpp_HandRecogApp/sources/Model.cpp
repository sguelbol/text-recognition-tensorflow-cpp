#include "../headers/Model.h"

#include <memory>
#include "../headers/DenseLayer.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/summary/summary_file_writer.h"
#include "../headers/Helper.h"
#include "tensorflow/cc/framework/gradients.h"


/**
 * @brief Constructs a Model object.
 *
 * This constructor initializes a Model object with the given model scope.
 *
 * @param modelScope The scope for the model.
 */
Model::Model(Scope &modelScope) : scope(modelScope), session(std::make_shared<ClientSession>(scope)) {
}

/**
 * @brief Model destructor.
 *
 * This is the destructor for the Model class.
 */
Model::~Model() = default;


/**
 * @brief Adds an input layer to the model.
 *
 * This function creates an input layer in the model with the specified input dimension which is the initial layer of
 * the model where the raw input data are fed. The `featuresDim` member variable stores the input dimension.
 * The `features` member variable is initialized with a `Placeholder` object, which represents the input features tensor.
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
 * This method adds a dense layer with the specified number of neurons and activation function to the model.
 * The input dimension for the layer is determined based on the output dimension of the previous layer.
 * If this is the first layer added to the model, the input dimension is set to the number of input features.
 * If this is the subsequent layer, the input dimension is set to the number of the dimensions of the neurons from the
 * previous layer which correspond to the output dimension of the previous layer.
 * The activation function determines the activation applied to the layer's output.
 *
 * @param numberOfNeurons The number of neurons in the dense layer.
 * @param activationFunction The activation function to apply to the layer's output.
 */
void Model::addDenseLayer(int numberOfNeurons, ActivationFunction activationFunction) {
    Scope denseLayerScope = scope.NewSubScope("Dense-Layer-" + std::to_string(layers.size()+1));
    std::unique_ptr<DenseLayer> denseLayer;
    int inputDim = *featuresDim;
    if (!this->layers.empty()) {
        auto &previousLayer = layers[layers.size() - 1];
        inputDim = previousLayer->getNumberOfNeurons();
    }
    *labelsDim = numberOfNeurons;
    denseLayer = std::make_unique<DenseLayer>(this->session, denseLayerScope, inputDim, numberOfNeurons, activationFunction);
    layers.push_back(std::move(denseLayer));
}


/**
 * @brief Builds the model.
 *
 * This method builds the model by getting the assembled operations of the forward pass of each dense layer in the model.
 * If it is the first layer, the `forward` method is called with `this->features` as the input. Otherwise, it is called with `this->model`
 * which holds the assembled operations for computation of the layers before.
 *
 * @see DenseLayer::initialForwardPass(shared_ptr<Placeholder> inputImage)
 * @see DenseLayer::subsequentForwardPass(Output previousLayerOutput)
 *
 * @note At least one dense layer muss be added to the model before calling this method.
 */
void Model::buildModel() {
    if (layers.empty()) {
        std::cerr << "No dense layers, at least one dense layer should be added to model!";
        std::exit(EXIT_FAILURE);
    }
    bool isFirstLayer = true;
    for (auto& layer : layers) {
        if (isFirstLayer) {
            this->model = layer->initialForwardPass(this->features);
            isFirstLayer = false;
        } else {
            this->model = layer->subsequentForwardPass(this->model);
        }
    }
}


/**
 * @brief Generates predictions using the model.
 *
 * This method takes in a `Tensor` object representing the input features and returns a `Tensor` object representing the model's predictions.
 *
 * @note the shape of `inputFeatures` must be [1, featruesDim], when it has shape of [featruesDim] it will be automatically reshaped
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
 *
 * @param inputFeatures The input features tensor.
 * @return The reshaped input features tensor or the original tensor if it has more than one dimension.
 */
Tensor Model::reshapeInput(Tensor inputFeatures) {
    if (inputFeatures.dims() == 1) {
        vector<Tensor> outputs;
        auto reshaped = Reshape(scope.NewSubScope("Reshape"), inputFeatures, {1, *featuresDim});
        session->Run({reshaped}, &outputs);
        return outputs[0];
    }
    return inputFeatures;
}

/**
 * @brief Trains the model on a written character.
 *
 * This method trains the model on a written character. This is useful to train wrong predicted characters.
 *
 * @param imageTensor The image tensor of the written character.
 * @param expectedNumber The expected number for the written character.
 */
void Model::trainOnWrittenChar(Tensor imageTensor, int expectedNumber) {
    Scope retrainScope = scope.NewSubScope("Training_On_Written_Char");
    auto onehot = OneHot(retrainScope, {expectedNumber}, Input::Initializer(10), Input::Initializer(1.0f), Input::Initializer(0.0f));
    vector<Tensor> output;
    session->Run({onehot}, &output);
    this->train(imageTensor, output[0], 10, 0.7f, 1);
}

/**
 * @brief Prints the details of the model.
 *
 * This function outputs the details of the model to the command-line. It prints the details of each layer in the model.
 * For each layer, the `printLayer()` method is called to print the details of that layer.
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
 * @brief Trains the model on the provided image and label tensors.
 *
 * This function trains the model using the provided image and label tensors for a given number of epochs.
 * The image and label tensors are divided into batches.
 * The operations for the backpropagation step is calculated using the `backpropagation` function,
 * which returns the operations as `Output` objects which computes the gradients and applies the updating of the
 * weights and biases.
 *
 * The training step is performed by calling the `Output` object returned by the `backpropagation` function
 * with the appropriate inputs and labels in tensorflow session. This step updates the model weights and biases based on the gradients
 * calculated during backpropagation.
 *
 * Finally, after all epochs are completed, a message is printed to the standard output stream indicating
 * the end of training.
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

    auto loss = Mean(lossScope.NewSubScope("Loss"), SquaredDifference(lossScope.WithOpName("Sigmoid-Cross-Entropy"), model, *this->labels), {0});
    std::vector<Output> apply_gradients = this->backpropagation(lossScope,learningRate, loss);

    int dataSize = imageBatches.dim_size(0);
    std::vector<Tensor> outputs;
    Tensor imageBatch;
    Tensor labelBatch;
    for (int i = 1; i <= maxEpochs; i++) {
        vector<Tensor> output1;
        for (int64_t num = 0; num < dataSize; num++) {
            imageBatch = imageBatches.SubSlice(num);
            labelBatch = labelBatches.SubSlice(num);
            //TODO: Only Batches of num % 8 = 0 allowed because of alignment error
            TF_CHECK_OK(session->Run({{*features, imageBatch}, {*this->labels, labelBatch}}, apply_gradients, {}, nullptr));
        }
        if (i % 10 == 0 || i == maxEpochs) {
            TF_CHECK_OK(session->Run({{*features, imageBatch}, {*this->labels, labelBatch}}, {loss}, &outputs));
            std::cout << "\nEpoch " << i << " Loss: " << outputs[0].flat<float>() << std::endl;

        }
    }
    std::cout << " " << std::endl;
    std::cout << "#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#" << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Training Time elapsed: " << elapsed.count() << "ms" << std::endl;

}

/**
 * @brief Computes backpropagation operation to update the weights and biases of the model.
 *
 * This function adds gradient nodes to the tensorflow graph to compute the symbolic partial derivative of the loss function
 * with respect to the weights and biases of all layers in the model.
 * These derivatives are stored in an 'Output' object named 'gradients'. This 'gradients' are utilized by the ApplyGradientDescent operation
 * for updading the weights and biases which improving the model and minimizing the loss during the training process.
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
 * @return vector of Variable containing the weights of all layers in the model.
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
 * @return A vector of Variable containing the biases of all layers in the model.
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
    Scope batchesScope = scope.NewSubScope("Batch-Generation");
    auto dataSetSize = images.dim_size(0);
    int numBatches = dataSetSize / batchSize;
    //calculates how many images will be used for the batches, a few images will be ignored which won't fit into a batch
    int dataForBatches = numBatches * batchSize;
    Placeholder imagePlc(batchesScope, DT_FLOAT, Placeholder::Shape({dataSetSize, -1}));
    auto sliced = Slice(batchesScope, imagePlc, {0, 0}, {dataForBatches, -1});
    auto flatten = Reshape(batchesScope, sliced, {-1});
    auto reshapeToBatches = Reshape(batchesScope, flatten, {numBatches, batchSize, -1});
    vector<Tensor> outputs;
    TF_CHECK_OK(session->Run({{imagePlc, images}}, {reshapeToBatches}, &outputs));
    batchImages = outputs[0];
    TF_CHECK_OK(session->Run({{imagePlc, labels}}, {reshapeToBatches}, &outputs));
    batchLabels = outputs[0];
    return std::make_tuple(batchImages, batchLabels);
}

/**
 * @brief Performs validation on the input image tensor and label tensor..
 *
 * This method calculates the predicted values for the input image tensor using the `predict` method.
 * The number of correct predictions is calculated by comparing the predicted class with the label class for each element of the predicted values.
 * The accuracy rate is printed in the command line.
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
