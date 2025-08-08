#include "tensor.h"
#include <cstdio>
#include <iostream>
#include <ostream>

int main() {
    try {
        // Test scalar (0D) tensor
        std::printf("Testing scalar tensor...\n");
        Tensor scalar({1}, Dtype::Float32);
        std::printf("%s", scalar.info().c_str());
        
        // Test vector (1D) tensor
        std::printf("\nTesting vector tensor...\n");
        Tensor vector({10}, Dtype::Float32);
        std::printf("%s", vector.info().c_str());
        
        // Test matrix (2D) tensor
        std::printf("\nTesting matrix tensor...\n");
        Tensor matrix({5, 5}, Dtype::Float32);
        std::printf("%s", matrix.info().c_str());
        
        // Test 3D tensor
        std::printf("\nTesting 3D tensor...\n");
        Tensor tensor3d({2, 3, 4}, Dtype::Float32);
        std::printf("%s", tensor3d.info().c_str());
        
        // Test 4D tensor (backward compatibility)
        std::printf("\nTesting 4D tensor (backward compatibility)...\n");
        Tensor tensor4d({2, 3, 4, 5}, Dtype::Float32);
        std::printf("%s", tensor4d.info().c_str());
        
        // Test shape accessors
        std::printf("\nTesting shape accessors...\n");
        std::printf("4D tensor shape: [");
        for (size_t i = 0; i < tensor4d.ndim(); ++i) {
            std::printf("%llu", tensor4d.shape()[i]);
            if (i < tensor4d.ndim() - 1) std::printf(", ");
        }
        std::printf("]\n");
        
        // Test detailed print
        std::cout << "\nTesting detailed print..." << std::endl;
        std::cout << matrix.info_full();
        
        std::printf("\nAll tests passed!\n");
        return 0;
    } catch (const std::exception& e) {
        std::printf("Error: %s\n", e.what());
        return 1;
    }
}