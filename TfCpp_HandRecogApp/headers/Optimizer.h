#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/ops.h>
#include <tensorflow/cc/framework/scope.h>
#include "tensorflow/cc/ops/standard_ops.h"


enum class OptimizerType {
    SGD,
    MOMENTUM
};

using namespace tensorflow;
using namespace tensorflow::ops;

/**
 * @class Optimizer
 * @brief The Optimizer class provides different optimization algorithms for training models.
 */
class Optimizer {

public:
    static Optimizer SGD(float learningRate);
    static Optimizer Momentum(float learningRate, float momentum);
    Output applyOptimizer(ClientSession &session, Scope &scope, Input weights, Input gradients, int inputDim, int numberOfNeurons);
    string getOptimizerType();

private:
    Optimizer(float learningRate);
    Optimizer(float learningRate, float momentum);
    OptimizerType type;
    float learningRate;
    float momentum;
};



#endif //OPTIMIZER_H
