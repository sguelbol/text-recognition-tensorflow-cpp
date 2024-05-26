#include "gtest/gtest.h"
#include "../headers/Model.h"
#include "../headers/Optimizer.h"
#include "../enum/ActivationFunction.h"

namespace {
    TEST(ModelTest, TestModel) {
        Scope scope = Scope::NewRootScope();
        Model mlp = Model(scope, Optimizer::SGD(0.9f));
        mlp.addInputLayer(2);
        mlp.addDenseLayer(2, ActivationFunction::SIGMOID);
        mlp.buildModel();


        Tensor features(DT_FLOAT, TensorShape({8, 2}));
        std::vector<std::vector<float>> featureData = {
            {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f},
            {0.0f, 0.0f},{0.0f, 1.0f},{1.0f, 0.0f},{1.0f, 1.0f}
        };
        auto tensorMap = features.tensor<float, 2>();
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 2; ++j) {
                tensorMap(i, j) = featureData[i][j];
            }
        }
        Tensor label(DT_FLOAT, TensorShape({8, 2}));
        std::vector<std::vector<float>> labelData = {
            {0.0f, 0.0f}, {0.0f, 1.0f}, {0.0f, 1.0f}, {1.0f, 1.0f},
            {0.0f, 0.0f}, {0.0f, 1.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}
        };
        auto tensorMap2 = label.tensor<float, 2>();
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 2; ++j) {
                tensorMap2(i, j) = labelData[i][j];
            }
        }

        ClientSession session(scope);
        mlp.train(features, label, 800, 8);
        auto sliceLabel1 = Slice(scope, features, {0, 0}, {1, -1});
        auto sliceLabel2 = Slice(scope, features, {1, 0}, {1, -1});
        auto sliceLabel3 = Slice(scope, features, {2, 0}, {1, -1});
        auto sliceLabel4 = Slice(scope, features, {3, 0}, {1, -1});
        vector<Tensor> output;
        session.Run({}, {sliceLabel1, sliceLabel2, sliceLabel3, sliceLabel4}, {}, &output);


        for (int i = 0; i < 4; i++) {
            Tensor prediction = mlp.predict(output[i]);
            std::cout << "Input: [" << features.matrix<float>()(i, 0) << ", "<< features.matrix<float>()(i, 1) << "]" << std::endl;
            for (int j = 0; j < prediction.dim_size(1); j++) {
                auto labelValue = label.matrix<float>()(i, j);
                auto predictedValue = prediction.flat<float>()(j);
                std::cout << " Label " << (j==0?"AND":"OR") << ": " << labelValue << " Predicted: " << predictedValue << std::endl;
                if (labelValue == 0) {
                    EXPECT_TRUE(predictedValue <= 0.1f);
                } else if (labelValue == 1) {
                    EXPECT_TRUE(predictedValue >= 0.9f);
                }
            }
        }
    }
}

