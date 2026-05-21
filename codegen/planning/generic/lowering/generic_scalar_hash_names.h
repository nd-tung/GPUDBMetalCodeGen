#pragma once

#include <string>
#include <utility>

namespace codegen {

struct ScalarCompositeHashNames {
    std::string map;

    std::string states() const { return map + "_states"; }
    std::string keys1() const { return map + "_keys1"; }
    std::string keys2() const { return map + "_keys2"; }
    std::string values() const { return map + "_values"; }
    std::string capacity() const { return "n_" + map; }
};

inline ScalarCompositeHashNames scalarCompositeHashNames(std::string map) {
    return ScalarCompositeHashNames{std::move(map)};
}

} // namespace codegen
