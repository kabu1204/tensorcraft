#include "tensor.h"
#include <cstdio>

int main() {
    // Test zeros
    auto zeros_tensor = Tensor::zeros({2, 3}, Dtype::Float32);
    std::printf("Zeros tensor:\n");
    std::printf("%s", zeros_tensor.info().c_str());
    
    // Test ones
    auto ones_tensor = Tensor::ones({2, 3}, Dtype::Float32);
    std::printf("\nOnes tensor:\n");
    std::printf("%s", ones_tensor.info().c_str());
    
    // Test arange
    auto arange_tensor = Tensor::arange(0, 10, 2, Dtype::Float32);
    std::printf("\nArange tensor (0, 10, 2):\n");
    std::printf("%s", arange_tensor.info().c_str());
    
    // Test arange with single parameter
    auto arange_single = Tensor::arange(5, Dtype::Float32);
    std::printf("\nArange tensor (5):\n");
    std::printf("%s", arange_single.info().c_str());
    
    return 0;
}