# Deploy a model to C++

!!! note "Draft"
    This guide is scaffolded. The outline below marks what it should cover.

[`TargetCpp`][manta.TargetCpp] lowers any Module to a typed Eigen C++
class over flat-C kernels, plus a CMake project.

```python
from manta import TargetCpp
TargetCpp(Sim(w), "out", class_name="Drone")
# → out/{drone.hpp, drone.cpp, drone_kernels.c/h, CMakeLists.txt}
```

## To cover

- What gets emitted (`.hpp` / `.cpp` / `_kernels.c/h` / `CMakeLists.txt`).
- The C++ surface mirrors numpy — the same `step` / `update` / `predict`
  / `control` loop, so a loop developed in Python ports verbatim.
- Building the CMake project and linking against Eigen.
- Which transforms lower (Sim, EKF with mutable state + Joseph, LQR).
- Feeding inputs and reading outputs across the ABI.

## Source material

- Reference: [Targets](../reference/targets.md)
- Code: `manta/codegen/cpp/`
- Concepts: [Codegen and backends](../explanation/codegen.md)
