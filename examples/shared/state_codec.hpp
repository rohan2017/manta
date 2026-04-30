#pragma once

#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "manta/core/craft.hpp"
#include "manta/geom/vec3.hpp"

namespace manta::examples {

// Minimal JSON encoder for craft state (position, orientation quaternion,
// linear & angular velocity in scene frame). Keeps the wire format trivial
// for the Python rerun viewer.
inline std::string encode_craft_state(double t_sec, const Craft& c) {
    auto p = c.scene_to_craft().position();
    auto q = c.scene_to_craft().orientation().raw();  // Eigen::Quaternion (w,x,y,z)
    auto v = c.scene_to_craft().vel_linear();
    auto w = c.scene_to_craft().vel_angular();

    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\"t\":%.6f,"
        "\"p\":[%.6f,%.6f,%.6f],"
        "\"q\":[%.6f,%.6f,%.6f,%.6f],"
        "\"v\":[%.6f,%.6f,%.6f],"
        "\"w\":[%.6f,%.6f,%.6f]}",
        t_sec,
        (double)p.x(), (double)p.y(), (double)p.z(),
        (double)q.w(), (double)q.x(), (double)q.y(), (double)q.z(),
        (double)v.x(), (double)v.y(), (double)v.z(),
        (double)w.x(), (double)w.y(), (double)w.z());
    return std::string(buf);
}

// Parse a list of N floats out of a tiny JSON array like "[0.1,0.2,0.3]".
// Returns true on success. Tolerant of whitespace; NOT a general JSON parser.
inline bool parse_float_array(std::string_view s, std::vector<float>& out, std::size_t expected) {
    out.clear();
    out.reserve(expected);
    auto lb = s.find('[');
    auto rb = s.rfind(']');
    if (lb == std::string_view::npos || rb == std::string_view::npos || rb <= lb) return false;
    std::string body(s.substr(lb + 1, rb - lb - 1));
    char* p = body.data();
    char* end = body.data() + body.size();
    while (p < end) {
        while (p < end && (*p == ' ' || *p == ',' || *p == '\t' || *p == '\n')) ++p;
        if (p >= end) break;
        char* next = nullptr;
        float v = std::strtof(p, &next);
        if (next == p) return false;
        out.push_back(v);
        p = next;
    }
    return out.size() == expected;
}

} // namespace manta::examples
