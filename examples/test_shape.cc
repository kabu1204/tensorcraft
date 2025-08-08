#include "tensor.h"
#include <iostream>

int main() {
    try {
        // Test scalar (0D) tensor
        std::cout << "Testing scalar tensor..." << std::endl;
        Tensor scalar({1}, Dtype::Float32);
        scalar.print("Scalar");
        
        // Test vector (1D) tensor
        std::cout << "\nTesting vector tensor..." << std::endl;
        Tensor vector({10}, Dtype::Float32);
        vector.print("Vector");
        
        // Test matrix (2D) tensor
        std::cout << "\nTesting matrix tensor..." << std::endl;
        Tensor matrix({5, 5}, Dtype::Float32);
        matrix.print("Matrix");
        
        // Test 3D tensor
        std::cout << "\nTesting 3D tensor..." << std::endl;
        Tensor tensor3d({2, 3, 4}, Dtype::Float32);
        tensor3d.print("3D Tensor");
        
        // Test 4D tensor (backward compatibility)
        std::cout << "\nTesting 4D tensor (backward compatibility)..." << std::endl;
        Tensor tensor4d({2, 3, 4, 5}, Dtype::Float32);
        tensor4d.print("4D Tensor");
        
        // Test shape accessors
        std::cout << "\nTesting shape accessors..." << std::endl;
        std::cout << "4D tensor shape: [";
        for (size_t i = 0; i < tensor4d.ndim(); ++i) {
            std::cout << tensor4d.shape()[i];
            if (i < tensor4d.ndim() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Test detailed print
        std::cout << "\nTesting detailed print..." << std::endl;
        matrix.print_detailed("Detailed Matrix");
        
        std::cout << "\nAll tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}