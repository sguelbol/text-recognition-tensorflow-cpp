#include "../headers/MNISTReader.h"

/**
 * @brief Checks if a file exists at the specified path.
 *
 * @param path The path to the file.
 * @return Returns true if the file exists, false otherwise.
 */
bool MNISTReader::fileExists(const string& path) {
    ifstream file(path.c_str());
    return file.good();
}

/**
 * @brief Read MNIST images from a given path
 *
 * This method reads MNIST images from the specified path and performs image preprocessing and normalization.
 * It returns the training set and testing set as Tensors.
 *
 * @param scope The scope to be used for running TensorFlow operations
 * @param path The path to the MNIST image dataset
 * @param trainingData The number of images to be used for training
 * @param validationData The number of images to be used for validation/testing
 *
 * @return A tuple of Tensors containing the training set and testing set of images
 */
tuple<Tensor, Tensor> MNISTReader::ReadMNISTImages(Scope &scope, string path, int trainingData, int validationData) {
    if (!fileExists(path)) {
        cerr << "Image dataset doesn't exists" << endl;
    }
    if (trainingData + validationData > 65000) {
        cerr << "Training data and Validation data overloads max available data num" << endl;

    }
    Scope imagePreprocessingScope = scope.NewSubScope("Training-Image-Preprocessing");
    Input  filename = Input::Initializer(path);
    auto file_reader = ReadFile(imagePreprocessingScope, filename);
    auto decodedPng = DecodePng(imagePreprocessingScope, file_reader, DecodePng::Channels(0));
    auto float_caster = Cast(imagePreprocessingScope, decodedPng, DT_FLOAT);
    auto reshaped = Reshape(imagePreprocessingScope, float_caster, {-1, 784});
    auto normalization = Div(imagePreprocessingScope, reshaped, {255.f});
    auto trainingSet = Slice(imagePreprocessingScope, normalization, {0, 0}, {trainingData, -1});
    auto testingSet = Slice(imagePreprocessingScope, normalization, {trainingData, 0}, {validationData, -1});
    ClientSession session(imagePreprocessingScope);
    vector<Tensor> outputs;
    TF_CHECK_OK(session.Run({trainingSet, testingSet}, &outputs));
    return make_tuple(outputs[0], outputs[1]);
}

/**
 * @brief Reads and preprocesses MNIST labels from files.
 *
 * This method reads the MNIST label dataset from the specified file path. It preprocesses the labels by performing the following steps:
 * 1. Reads the file contents as bytes.
 * 2. Decodes the raw bytes as signed 8-bit integers.
 * 3. Casts the data to floating-point numbers.
 * 4. Reshapes the data into the desired shape.
 * 5. Slices the training and testing datasets.
 *
 * @param scope The TensorFlow scope to use for running operations.
 * @param path The file path to the MNIST label dataset.
 * @param trainingData The number of training samples.
 * @param validationData The number of testing samples.
 *
 * @return A tuple containing the training and testing label tensors.
 */
tuple<Tensor, Tensor> MNISTReader::ReadMNISTLabels(Scope &scope, string path, int trainingData, int validationData) {
    if (!fileExists(path)) {
        cerr << "Label dataset doesn't exists" << endl;
    }
    Scope labelPreprocessingScope = scope.NewSubScope("Training-Label-Preprocessing");
    Input bytes = Input::Initializer(path);
    auto file_reader = ReadFile(labelPreprocessingScope, bytes);
    auto decodedRaw = DecodeRaw(labelPreprocessingScope, file_reader, DT_INT8);
    auto float_caster = Cast(labelPreprocessingScope, decodedRaw, DT_FLOAT);
    auto reshaped = Reshape(labelPreprocessingScope, float_caster, {-1, 10});
    auto trainingSet = Slice(labelPreprocessingScope, reshaped, {0, 0}, {trainingData, -1});
    auto testingSet = Slice(labelPreprocessingScope, reshaped, {trainingData, 0}, {validationData, -1});
    ClientSession session(labelPreprocessingScope);
    vector<Tensor> outputs;
    session.Run({trainingSet, testingSet}, &outputs);
    return make_tuple(outputs[0], outputs[1]);
}

