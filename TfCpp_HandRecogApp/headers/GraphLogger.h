#include "tensorflow/cc/ops/standard_ops.h"

#ifndef MULTILAYERPERCEPTRON_GRAPHLOGGER_H
#define MULTILAYERPERCEPTRON_GRAPHLOGGER_H

using namespace std;
using namespace tensorflow;

class GraphLogger {
public:
    static void logGraph(Scope& scope);
};
#endif //MULTILAYERPERCEPTRON_GRAPHLOGGER_H
