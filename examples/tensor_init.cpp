#include "tensor.h"
#include <iostream>

int main() {
    // Test zeros
    auto zeros_tensor = Tensor::zeros({2, 3}, Dtype::Float32);
    std::cout << "Zeros tensor:\n";
    zeros_tensor.print("zeros");
    
    // Test ones
    auto ones_tensor = Tensor::ones({2, 3}, Dtype::Float32);
    std::cout << "\nOnes tensor:\n";
    ones_tensor.print("ones");
    
    // Test arange
    auto arange_tensor = Tensor::arange(0, 10, 2, Dtype::Float32);
    std::cout << "\nArange tensor (0, 10, 2):\n";
    arange_tensor.print("arange");
    
    // Test arange with single parameter
    auto arange_single = Tensor::arange(5, Dtype::Float32);
    std::cout << "\nArange tensor (5):\n";
    arange_single.print("arange_single");
    
    return 0;
}