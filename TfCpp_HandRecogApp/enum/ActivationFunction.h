#ifndef ACTIVATIONFUNCTION_H
#define ACTIVATIONFUNCTION_H

enum class ActivationFunction {
    SIGMOID, RELU, SOFTMAX
};

class ActivationnFunctionConverter {
public:
    static std::string toString(ActivationFunction activationFunction) {
        switch(activationFunction) {
            case ActivationFunction::SIGMOID: return "Sigmoid";
            case ActivationFunction::RELU:    return "Relu";
            case ActivationFunction::SOFTMAX: return "Softmax";
            default:                          return "Unknown";
        }
    }
};

#endif //ACTIVATIONFUNCTION_H
