#include "../headers/Optimizer.h"

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/ops/const_op.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/cc/ops/state_ops.h>
#include <tensorflow/cc/ops/training_ops.h>

using namespace tensorflow::ops;

/**
 * @brief Constructs a Optimizer object.
 *
 * This constructor initializes an Optimizer object with the given learning rate for SGD optimization.
 * The type of the optimizer is set to SGD (Stochastic Gradient Descent).
 *
 * @param learningRate The learning rate to be used by the optimizer.
 */
Optimizer::Optimizer(float learningRate) : type(OptimizerType::SGD), learningRate(learningRate) {}


/**
 * @brief Constructs a Optimizer object.
 *
 * This constructor initializes an Optimizer object with the given learning rate and the momentum for Momentum optimization.
 * The type of the optimizer is set to Momentum.
 *
 * @param learningRate The learning rate to be used by the optimizer.
 * @param momentum The momentum to be used by the optimizer.
 */
Optimizer::Optimizer(float learningRate, float momentum) : type(OptimizerType::MOMENTUM), learningRate(learningRate), momentum(momentum) {}


/**
 * @brief Create a Stochastic Gradient Descent (SGD) optimizer object with the given learning rate.
 *
 * @param learningRate The learning rate for the SGD optimizer.
 * @return An Optimizer object initialized with SGD type and the given learning rate.
 */
Optimizer Optimizer::SGD(float learningRate) {
    return Optimizer(learningRate);
}

/**
  * @brief Create a Momentum optimizer object with the given learning rate and the momentum.
 *
 * @param learningRate The learning rate value for the momentum optimizer.
 * @param momentum The momentum value for the momentum optimizer.
 * @return An Optimizer object initialized with Momentum type, given learning rate and the momentum.
 */
Optimizer Optimizer::Momentum(float learningRate, float momentum) {
    return Optimizer(learningRate, momentum);
}

/**
 * Apply the optimizer to update the weights
 *
 * @param session the client session for running the TensorFlow graph
 * @param scope the scope of the optimizer
 * @param weights the input weights to be updated
 * @param gradients the gradients used for updating the weights
 * @param inputDim the input dimension for the optimizer
 * @param numberOfNeurons the number of neurons for the optimizer
 * @return the output of applying the optimizer
 */
Output Optimizer::applyOptimizer(ClientSession &session, Scope &scope, Input weights, Input gradients, int inputDim, int numberOfNeurons) {
    Output appliedOtimizer;
    switch (type) {
        case OptimizerType::SGD:
            std::cout << &weights << std::endl;
            appliedOtimizer = ApplyGradientDescent(scope, weights, Cast(scope, learningRate,  DT_FLOAT), gradients);
            break;
        case OptimizerType::MOMENTUM:
            Input accum = Variable(scope, {inputDim, numberOfNeurons}, DT_FLOAT);
            auto zeros = ZerosLike(scope, weights);
            auto initAccum = Assign(scope, accum, zeros);
            TF_CHECK_OK(session.Run({initAccum}, nullptr));
            appliedOtimizer = ApplyMomentum(scope, weights, accum, Cast(scope, learningRate,  DT_FLOAT), gradients, Const(scope, momentum));
            break;

    }
    return appliedOtimizer;
}

/**
 * Get the optimizer type as a string.
 *
 * @return The optimizer type as a string.
 */
string Optimizer::getOptimizerType() {
    switch (type) {
        case OptimizerType::SGD: return "SGD"; break;
        case OptimizerType::MOMENTUM: return "Momentum"; break;
    }
}


