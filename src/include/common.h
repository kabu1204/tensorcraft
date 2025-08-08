#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <type_traits>
#include <string>
#include <stdexcept>
#include <limits>

// Assertion macro that throws an exception
#define CHECK_THROW(condition) \
    do { \
        if (!(condition)) { \
            throw std::runtime_error("Assertion failed: " #condition); \
        } \
    } while (0)


enum class Dtype {
    Float32 = 0,
    Float16 = 1,
    BFloat16 = 2,
    Int8 = 3,
    Int16 = 4,
    Int32 = 5,
    Int64 = 6,
    UInt8 = 7,
    UInt16 = 8,
    UInt32 = 9,
    UInt64 = 10,
    
    Count  // Number of data types, used for validation
};

// Helper function to get the size of a data type in bytes
inline size_t dtype_size(Dtype dtype) {
    switch (dtype) {
        case Dtype::Float32: return sizeof(float);
        case Dtype::Float16: return 2;  // 16-bit float
        case Dtype::BFloat16: return 2; // 16-bit bfloat
        case Dtype::Int8: return sizeof(int8_t);
        case Dtype::Int16: return sizeof(int16_t);
        case Dtype::Int32: return sizeof(int32_t);
        case Dtype::Int64: return sizeof(int64_t);
        case Dtype::UInt8: return sizeof(uint8_t);
        case Dtype::UInt16: return sizeof(uint16_t);
        case Dtype::UInt32: return sizeof(uint32_t);
        case Dtype::UInt64: return sizeof(uint64_t);
        default: throw std::runtime_error("Unknown data type");
    }
}

// Helper function to get the name of a data type as string
inline const char* dtype_name(Dtype dtype) {
    switch (dtype) {
        case Dtype::Float32: return "float32";
        case Dtype::Float16: return "float16";
        case Dtype::BFloat16: return "bfloat16";
        case Dtype::Int8: return "int8";
        case Dtype::Int16: return "int16";
        case Dtype::Int32: return "int32";
        case Dtype::Int64: return "int64";
        case Dtype::UInt8: return "uint8";
        case Dtype::UInt16: return "uint16";
        case Dtype::UInt32: return "uint32";
        case Dtype::UInt64: return "uint64";
        default: return "unknown";
    }
}


struct Stride {
    std::array<uint32_t, 4> dims;
    
    Stride() { dims = {0, 0, 0, 0}; }
    Stride(uint32_t b, uint32_t c, uint32_t h, uint32_t w) {
        dims[0] = b;
        dims[1] = c;
        dims[2] = h;
        dims[3] = w;
    }
    
    uint32_t& operator[](size_t i) { return dims[i]; }
    const uint32_t& operator[](size_t i) const { return dims[i]; }
    
    bool operator==(const Stride& other) const { return dims == other.dims; }
    bool operator!=(const Stride& other) const { return dims != other.dims; }
};
