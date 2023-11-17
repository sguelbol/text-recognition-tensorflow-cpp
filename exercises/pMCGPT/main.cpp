#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/public/session_options.h"

using namespace tensorflow;

int main() {
    // Define training data
    std::vector<float> features = {0.1, 0.2};
    float target = 0.3;

    // Define perceptron weights and bias
    float weight1 = 0.1;
    float weight2 = 0.2;
    float bias = 0.0;

    // Create a TensorFlow session
    Session* session;
    SessionOptions session_options;
    NewSession(session_options, &session);

    // Define the computational graph
    Scope root = Scope::NewRootScope();
    Output features_tensor = Placeholder(root, DT_FLOAT);
    Output target_tensor = Placeholder(root, DT_FLOAT);

    Output weights_tensor1 = Const(root, {{weight1}});
    Output weights_tensor2 = Const(root, {{weight2}});
    Output bias_tensor = Const(root, {{bias}});

    Output perceptron = Add(
            root.WithOpName("perceptron"),
            Add(
                    root.WithOpName("weighted_sum"),
                    Mul(root, features_tensor, weights_tensor1),
                    Mul(root, features_tensor, weights_tensor2)
            ),
            bias_tensor
    );

    Output loss = Square(
            root.WithOpName("loss"),
            Subtract(root, perceptron, target_tensor)
    );

    // Define the optimizer and training operation
    Output learning_rate_tensor = Const(root, {{0.01}});
    Output optimizer = ApplyGradientDescent(
            root.WithOpName("optimizer"),
            weights_tensor1,
            learning_rate_tensor,
            Reshape(root, GradientDescent(root, loss, weights_tensor1), {}),
            NoOutputs({loss.op()}));
    Output optimizer2 = ApplyGradientDescent(
            root.WithOpName("optimizer2"),
            weights_tensor2,
            learning_rate_tensor,
            Reshape(root, GradientDescent(root, loss, weights_tensor2), {}),
            NoOutputs({loss.op()}));

    // Initialize variables
    std::vector<Output> initialize_ops;
    TF_CHECK_OK(session->Run({{features_tensor, {{features[0], features[1]}}}}, {}, {"perceptron"}, nullptr));
    TF_CHECK_OK(session->Run({}, {}, {"optimizer", "optimizer2", "weighted_sum", "loss"}, &initialize_ops));

    // Training loop
    const int num_epochs = 1000;
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::vector<Output> outputs;
        TF_CHECK_OK(session->Run({{target_tensor, {{target}}}}, {"optimizer", "optimizer2", "loss"}, {}, &outputs));
        float current_loss = outputs[2].scalar<float>()(0);
        if (epoch % 100 == 0) {
            printf("Epoch %d - Loss: %f\n", epoch, current_loss);
        }
    }

    // Print the trained weights and bias
    std::vector<Output> trained_weights, trained_bias;
    TF_CHECK_OK(session->Run({}, {"weighted_sum"}, {}, &trained_weights));
    TF_CHECK_OK(session->Run({}, {"perceptron"}, {}, &trained_bias));
    printf("Trained weights: %f, %f\n", trained_weights[0].scalar<float>()(0), trained_weights[1].scalar<float>()(0));
    printf("Trained bias: %f\n", trained_bias[0].scalar<float>()(0));

    // Close the session
    session->Close();
    return 0;
}
