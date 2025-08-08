#include "tensor.h"
#include <cstdio>

int main() {
    try {
        std::printf("=== Tensor Creation Example ===\n");
        
        // Test Stride creation (should work)
        Stride stride(24, 12, 4, 1);
        std::printf("Stride created: [%u, %u, %u, %u]\n", stride[0], stride[1], stride[2], stride[3]);
        
        // Test basic tensor creation
        Tensor tensor({1, 1, 2, 2});
        // tensor.fill(1.0);
        std::printf("%s", tensor.info().c_str());
        
        std::printf("Example completed successfully!\n");
        return 0;
    } catch (const std::exception& e) {
        std::printf("Error: %s\n", e.what());
        return 1;
    }
}