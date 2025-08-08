#include "tensor.h"
#include "slice.h"
#include <cstdio>

int main() {
    // Create a 2D tensor 3x4 and fill with arange
    auto t = Tensor::arange(12, Dtype::Float32).view({3,4}, {4,1}, 0);
    std::printf("%s", t.info().c_str());

    // Element access
    std::printf("t(1,2) = %g\n", static_cast<double>(t.at<float>(1,2)));

    // Select row 1 -> shape [4]
    auto row1 = t.select(0, 1);
    std::printf("%s", row1.info().c_str());

    // Narrow columns [1,3)
    auto cols = t.narrow(1, 1, 2);
    std::printf("%s", cols.info().c_str());

    // Element access
    std::printf("cols(1,1) = %g\n", static_cast<double>(cols.at<float>(1,1)));

    // Slice rows [0,3) step 2 and cols [0,4) step 2 => shape [2,2]
    auto even_rc = t.slice({Slice::range(0,3,2), Slice::range(0,4,2)});
    std::printf("%s", even_rc.info().c_str());

    // operator[] chain: t[1][Slice::range(1,4,2)] -> shape [2]
    auto chain = t[1][Slice::range(1,4,2)];
    std::printf("%s", chain.info_full().c_str());

    // initializer_list multi-index: t[{1, {0,2}}]
    int n = 2;
    auto ilist = t[{1, {0,n}}];
    std::printf("%s", ilist.info().c_str());

    auto ilist2 = t[{{0,3}, 0}];
    std::printf("%s", ilist2.info().c_str());

    return 0;
}


