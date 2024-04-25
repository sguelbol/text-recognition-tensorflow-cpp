#include  <iostream>
#include <tensorflow/core/framework/tensor.h>


int main() {
    std::cout << "Hello, World!" << std::endl;

    // Scalar
    tensorflow::Tensor a(tensorflow::DT_FLOAT, tensorflow::TensorShape({1}));
    a.scalar<float>()() = 0.9;
    a.scalar<float>()() = 0.5;
    //std::cout << a << std::endl;

    // Vector tensor 1
    tensorflow::Tensor b(tensorflow::DT_FLOAT, tensorflow::TensorShape({4}));
    auto b_vec = b.vec<float>();
    b.vec<float>()(0) = 1.0f;
    b_vec(1) = 2.0f;
    b_vec(2) = 3.2f;
    //std::cout << b << std::endl;
    //std::cout << b_vec << std::endl;


    // Vector tensor 2
    std::vector<float> array_data = {1.3f, 2.0, 3.0, 4.0};
    tensorflow::Tensor tensorVector(tensorflow::DT_FLOAT, tensorflow::TensorShape{4});
    auto tensor_data = tensorVector.vec<float>();
    std::copy(array_data.begin(), array_data.end(), tensor_data.data());
    //std::cout << tensor_data << std::endl;

    // Matrix tensor 3
    std::vector<std::vector<float>> input_data = {{1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({4, 3}));
    float *tensor_data3 = input_tensor.flat<float>().data();
    for (int x = 0; x < 4; ++x) {
        for (int y = 0; y < 3; ++y) {
            tensor_data3[3 * x + y] = input_data[x][y];
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << tensor_data3[i * 3 + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << input_tensor.DebugString() << std::endl;


    // Matrix tensor 4
    tensorflow::Tensor matrix(tensorflow::DT_FLOAT, tensorflow::TensorShape({4, 3}));
    auto matrix_data = matrix.tensor<float, 2>();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            matrix_data(i, j) = input_data[i][j];
        }
    }
    std::cout << matrix_data << std::endl;
    std::cout << matrix.DebugString() << std::endl;


    // Scalar
    tensorflow::Tensor scalar_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1}));
    auto scalar_data = scalar_tensor.scalar<float>();
    scalar_data() = 3.2f;
    std::cout << scalar_tensor.DebugString() << std::endl;



    return 0;
}
