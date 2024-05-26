#include "tensorflow/cc/ops/standard_ops.h"

#ifndef MULTILAYERPERCEPTRON_GRAPHLOGGER_H
#define MULTILAYERPERCEPTRON_GRAPHLOGGER_H

using namespace std;
using namespace tensorflow;

/**
 * @class GraphLogger
 * @brief The GraphLogger class is responsible for logging the tensorflow graph for visualization usage in tensorboard.
 */
class GraphLogger {
public:
    static void logGraph(Scope& scope);
};
#endif //MULTILAYERPERCEPTRON_GRAPHLOGGER_H
