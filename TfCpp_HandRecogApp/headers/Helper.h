#undef signals
#define singals tf_signals
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#undef signals

#ifndef HELPER_H
#define HELPER_H

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

class Helper {

public:

    static void printImageInConsole(Tensor &image);
    static void printLabelInConsole(Tensor &label);
    static Tensor calculatePredictedClass(Tensor &modelOutput);
};



#endif //HELPER_H
