#include "common.h"
#include "tensor.h"
#include "slice.h"
#include <cstdio>
#include <exception>

static void test_transpose_2d() {
    auto base = Tensor::arange(6, Dtype::Float32);
    auto a = base.view({2,3}, {3,1}, 0);

    CHECK_THROW(a.at<float>(0,0) == 0.0f);
    CHECK_THROW(a.at<float>(1,2) == 5.0f);

    auto t = a.T();
    CHECK_THROW(t.shape().size() == 2);
    CHECK_THROW(t.shape()[0] == 3 && t.shape()[1] == 2);
    CHECK_THROW(t.at<float>(2,1) == 5.0f);
}

static void test_permute_3d() {
    auto base = Tensor::arange(24, Dtype::Float32);
    auto a = base.view({2,3,4}, {12,4,1}, 0);

    CHECK_THROW(a.at<float>(1,2,3) == 1.0f*12 + 2.0f*4 + 3.0f);

    auto p = a.permute({1,0,2}); // shape [3,2,4]
    CHECK_THROW(p.shape().size() == 3);
    CHECK_THROW(p.shape()[0] == 3 && p.shape()[1] == 2 && p.shape()[2] == 4);

    // after permute, p(j,i,k) should equal a(i,j,k)
    CHECK_THROW(p.at<float>(2,1,3) == a.at<float>(1,2,3));
    CHECK_THROW(p.at<float>(0,0,1) == a.at<float>(0,0,1));
}

static void test_reshape_contiguous() {
    auto base = Tensor::arange(6, Dtype::Float32);
    auto a = base.view({2,3}, {3,1}, 0); // contiguous

    auto r = a.reshape({3,2});
    CHECK_THROW(r.shape().size() == 2);
    CHECK_THROW(r.shape()[0] == 3 && r.shape()[1] == 2);
    std::printf("%s", r.info().c_str());
    // flat order preserved
    CHECK_THROW(r.at<float>(0,0) == 0.0f);
    CHECK_THROW(r.at<float>(2,1) == 5.0f);
}

static void test_reshape_noncontiguous_throws() {
    auto base = Tensor::arange(12, Dtype::Float32);
    auto a = base.view({3,4}, {4,1}, 0);
    auto nc = a.slice(1, 0, 4, 2); // non-contiguous columns
    bool threw = false;
    try {
        auto r = nc.reshape({2,3});
        (void)r;
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK_THROW(threw);
}

static void test_contiguous_enables_reshape() {
    auto base = Tensor::arange(12, Dtype::Float32);
    auto a = base.view({3,4}, {4,1}, 0);
    auto nc = a.slice(1, 0, 4, 2); // non-contiguous columns -> shape [3,2]
    bool threw = false;
    try {
        auto r = nc.reshape({2,3});
        (void)r;
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK_THROW(threw);

    auto c = nc.contiguous();
    auto r2 = c.reshape({2,3});
    CHECK_THROW(r2.shape()[0] == 2 && r2.shape()[1] == 3);
    CHECK_THROW(nc.at<float>(0,0) == c.at<float>(0,0));
    CHECK_THROW(nc.at<float>(2,1) == c.at<float>(2,1));
}

static void test_unified_iterator_write() {
    // creation
    auto t = Tensor::zeros({2,3}, Dtype::Float32);
    float val = 1.5f;
    for (auto it = t.begin(); it != t.end(); ++it) {
        (*it).as_float32() = val;
        val += 1.0f;
    }
    // verify
    float expect = 1.5f;
    for (uint64_t i = 0; i < t.shape()[0]; ++i) {
        for (uint64_t j = 0; j < t.shape()[1]; ++j) {
            CHECK_THROW(std::fabs(t.at<float>(i,j) - expect) < 1e-6f);
            expect += 1.0f;
        }
    }
}

int main() {
    try {
        test_transpose_2d();
        test_permute_3d();
        test_reshape_contiguous();
        test_reshape_noncontiguous_throws();
        test_contiguous_enables_reshape();
        test_unified_iterator_write();
        std::printf("All tensor view ops tests passed.\n");
        return 0;
    } catch (const std::exception& e) {
        std::printf("Test failed: %s\n", e.what());
        return 1;
    }
}


