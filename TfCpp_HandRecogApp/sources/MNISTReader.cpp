#include "../headers/MNISTReader.h"

bool MNISTReader::fileExists(const string& path) {
    ifstream file(path.c_str());
    return file.good();
}

tuple<Tensor, Tensor> MNISTReader::ReadMNISTImagesWithTF(Scope &scope, string path, int trainingData, int validationData) {
    if (!fileExists(path)) {
        cerr << "Image dataset doesn't exists" << endl;
    }
    if (trainingData + validationData > 65000) {
        cerr << "Training data and Validation data overloads max available data num" << endl;

    }
    Scope scope2 = scope.NewSubScope("MN1");
    Input  filename = Input::Initializer(path);
    auto file_reader = ReadFile(scope2, filename);
    auto decodedPng = DecodePng(scope2, file_reader, DecodePng::Channels(0));
    auto float_caster = Cast(scope2, decodedPng, DT_FLOAT);
    auto reshaped = Reshape(scope2, float_caster, {-1, 784});
    auto trainingSet = Div(scope, Slice(scope2, reshaped, {0, 0}, {trainingData, -1}), {255.f});
    auto testingSet = Div(scope, Slice(scope2, reshaped, {trainingData, 0}, {validationData, -1}), {255.f});

    ClientSession session(scope2);
    vector<Tensor> outputs;
    TF_CHECK_OK(session.Run({trainingSet, testingSet}, &outputs));
    return make_tuple(outputs[0], outputs[1]);
}

tuple<Tensor, Tensor> MNISTReader::ReadMNISTLabelsWithTF(Scope &scope, string path, int trainingData, int validationData) {
    if (!fileExists(path)) {
        cerr << "Label dataset doesn't exists" << endl;
    }
    Scope scope2 = scope.NewSubScope("MN2");
    Input bytes = Input::Initializer(path);
    auto file_reader = ReadFile(scope2, bytes);
    auto decodedRaw = DecodeRaw(scope2, file_reader, DT_INT8);
    auto float_caster = Cast(scope2, decodedRaw, DT_FLOAT);
    auto reshaped = Reshape(scope2, float_caster, {-1, 10});
    auto trainingSet = Slice(scope2, reshaped, {0, 0}, {trainingData, -1});
    auto testingSet = Slice(scope2, reshaped, {trainingData, 0}, {validationData, -1});
    ClientSession session(scope2);
    vector<Tensor> outputs;
    session.Run({trainingSet, testingSet}, &outputs);
    return make_tuple(outputs[0], outputs[1]);
}

