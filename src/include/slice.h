#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <variant>
#include <initializer_list>

struct Slice {
    int64_t start;
    int64_t end;
    int64_t step;

    static constexpr int64_t kNegInf = std::numeric_limits<int64_t>::min();
    static constexpr int64_t kPosInf = std::numeric_limits<int64_t>::max();

    // for brace-init like {3,5}
    constexpr Slice() : start(kNegInf), end(kPosInf), step(1) {}
    constexpr Slice(int64_t s) : start(s), end(s + 1), step(1) {}
    constexpr Slice(int64_t s, int64_t e) : start(s), end(e), step(1) {}
    constexpr Slice(int64_t s, int64_t e, int64_t st) : start(s), end(e), step(st) {}

    static Slice all() { return {kNegInf, kPosInf, 1}; }
    static Slice range(int64_t s, int64_t e, int64_t st = 1) { return {s, e, st}; }
};

using Index = std::variant<int64_t, Slice>;
inline Slice S(int64_t s, int64_t e, int64_t st = 1) { return Slice::range(s, e, st); }
inline constexpr Slice All = {Slice::kNegInf, Slice::kPosInf, 1};

struct IndexArg {
    Index value;

    IndexArg(int64_t i) : value(i) {}
    IndexArg(Slice s) : value(s) {}
    IndexArg(std::initializer_list<int64_t> ilist) {
        if (ilist.size() == 1) {
            value = Slice(*ilist.begin());
        } else if (ilist.size() == 2) {
            auto it = ilist.begin();
            value = Slice{*it, *(it + 1)}; // step defaults to 1
        } else if (ilist.size() == 3) {
            auto it = ilist.begin();
            value = Slice{*it, *(it + 1), *(it + 2)};
        } else {
            throw std::runtime_error("brace-init: too many arguments");
        }
    }
    operator Index() const { return value; }
};


