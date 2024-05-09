#include "../headers/Helper.h"
#include <tensorflow/cc/client/client_session.h>

/**
 * @brief Prints the image tensor in the console.
 *
 * This method prints the shape of the image tensor and the values of each pixel in the console.
 *
 * @param image The image tensor to be printed.
 */
void Helper::printImageInConsole(Tensor &image) {
    std::cout << image.shape() << std::endl;
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; ++j) {
            std::cout << image.flat<float>()(28*i+j) << " ";
        }
        std::cout << std::endl;
    }
}

/**
 * @brief Prints the label tensor in the console.
 *
 * This method prints the label tensor in the console.
 *
 * @param label The label tensor to be printed.
 */
void Helper::printLabelInConsole(Tensor &label) {
    std::cout << "Label: " << std::endl;
    std::cout << label.flat<float>() << std::endl;
}

/**
 * \brief Calculates the predicted class for a given model output.
 *
 * This method takes a model output tensor and returns the predicted class based on the largest value in the tensor.
 *
 * Example:
 * modelOutput = [4, 61, 21, 3, 13] => returns 2 (here modelOutput[2] is the largest element)
 *
 * \param modelOutput The predicted output tensor of the model.
 *
 * \return The predicted class as tensor.
 */
Tensor Helper::calculatePredictedClass(Tensor &modelOutput) {
    Scope scope = Scope::NewRootScope();
    ClientSession session(scope);
    auto predictedClass = ArgMax(scope, modelOutput, 1);
    vector<Tensor> output;
    session.Run({predictedClass}, &output);
    return output[0];
}


