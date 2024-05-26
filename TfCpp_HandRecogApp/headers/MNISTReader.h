#include <fstream>
#include "tensorflow/cc/ops/standard_ops.h"
#include <tensorflow/cc/client/client_session.h>

#ifndef MULTILAYERPERCEPTRON_MNISTREADER_H
#define MULTILAYERPERCEPTRON_MNISTREADER_H

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

/**
 * @class MNISTReader
 * @brief A utility class that  reads the MNIST data and loads it into a tensor.
 *
 * */
class MNISTReader {
public:
    static bool fileExists(const string& path);
    static tuple<Tensor, Tensor> ReadMNISTImages(Scope &scope, string path, int trainingData, int validationData);
    static tuple<Tensor, Tensor> ReadMNISTLabels(Scope &scope, string path, int trainingData, int validationData);
};


#endif //MULTILAYERPERCEPTRON_MNISTREADER_H
