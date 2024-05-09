#ifndef ACTIVATIONFUNCTION_H
#define ACTIVATIONFUNCTION_H

enum class ActivationFunction {
    SIGMOID, RELU, SOFTMAX, SELU, ELU
};


/**
 * @class ActivationFunctionConverter
 * @brief A utility class for converting an ActivationFunction enum to a string representation
 */
class ActivationFunctionConverter {
    /**
     * @brief Converts the given ActivationFunction enum value to a string representation.
     *        Possible values are: SIGMOID, RELU, SOFTMAX, SELU, ELU
     * @param activationFunction The ActivationFunction enum value
     * @return The string representation of the activation function
     */
public:
    static std::string toString(ActivationFunction activationFunction) {
        switch(activationFunction) {
            case ActivationFunction::SIGMOID: return "Sigmoid";
            case ActivationFunction::RELU:    return "Relu";
            case ActivationFunction::SOFTMAX: return "Softmax";
            case ActivationFunction::SELU: return "Selu";
            case ActivationFunction::ELU: return "Elu";
            default: return "Unknown";
        }
    }
};

#endif //ACTIVATIONFUNCTION_H
