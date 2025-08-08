#include "tensor.h"
#include <iostream>

int main() {
    try {
        std::cout << "=== Tensor Creation Example ===" << std::endl;
        
        // Test Stride creation (should work)
        Stride stride(24, 12, 4, 1);
        std::cout << "Stride created: [" << stride[0] << ", " << stride[1] 
                  << ", " << stride[2] << ", " << stride[3] << "]" << std::endl;
        
        // Test basic tensor creation
        Tensor tensor({1, 1, 2, 2});
        // tensor.fill(1.0);
        tensor.print("Test Tensor");
        
        std::cout << "Example completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}