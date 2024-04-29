#include <fstream>
#include <filesystem>
#define quint8 tf_quint8
#define qint8 tf_qint8
#define quint16 tf_quint16
#define qint16 tf_qint16
#define qint32 tf_qint32

#include "headers/Model.h"
#include "headers/Helper.h"
#include "headers/MNISTReader.h"
#include "headers/GraphLogger.h"
#include "tensorflow/core/framework/tensor.h"
#include "enum/ActivationFunction.h"
#undef quint8
#undef qint8
#undef quint16
#undef qint16
#undef qint32

#include "MainWindow.h"
#include <QApplication>

using namespace std;

int main(int argc, char *argv[]) {
    Scope scope = Scope::NewRootScope();
    Model mlp = Model(scope);
    mlp.addInputLayer(784);
    mlp.addDenseLayer(256, ActivationFunction::SOFTMAX);
    mlp.addDenseLayer(128, ActivationFunction::RELU);
    mlp.addDenseLayer(10, ActivationFunction::SOFTMAX);
    mlp.buildModel();
    mlp.printModel();

    //Read MNIST
    string const path = filesystem::current_path().parent_path().generic_string();
    string const pathTrainingsImages = path + "/dataset/mnist_images.png";
    string const pathTrainingsLabels = path + "/dataset/mnist_labels_uint8";
    Tensor trainingImages, testingImages;
    tie(trainingImages, testingImages) = MNISTReader::ReadMNISTImagesWithTF(scope, pathTrainingsImages, 40000, 20000);
    Tensor trainingLabels, testingLabels;
    tie(trainingLabels, testingLabels) = MNISTReader::ReadMNISTLabelsWithTF(scope, pathTrainingsLabels, 40000, 20000);



    mlp.train(trainingImages, trainingLabels, 20, 0.5f, 64);
    mlp.validate(testingImages, testingLabels);

    ClientSession session(scope);
    Tensor labelMNIST1 = testingLabels.SubSlice(16);
    Helper::printLabelInConsole(labelMNIST1);


    Tensor img = testingImages.SubSlice(16);
    Tensor tf = mlp.predict(img);
    Tensor lk = Helper::calculatePredictedClass(tf);


    GraphLogger::logGraph(scope);
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
