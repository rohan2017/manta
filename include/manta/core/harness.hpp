#pragma once

namespace manta {

// Harness — the polymorphic base class for everything that drives a
// `WorldT<Scalar>` (or composes other harnesses).
//
// The codegen produces a per-Target Harness subclass:
//
//   * World-only Target (e.g. ex1, ex4, ex7, connect_demo):
//         struct manta_gen::ex1::Harness : public manta::Harness {
//             void setup()    override { manta_gen::ex1::setup();    }
//             void tick()     override { manta_gen::ex1::tick();     }
//             void shutdown() override { manta_gen::ex1::shutdown(); }
//         };
//         extern Harness harness;
//
//   * Filter Target (ex6, ukf_smoke): same shape, delegating to the
//     filter harness's free functions (which run predict + update_n
//     instead of w.update()).
//
//   * Composed Target (ex5, ex8): the composed harness's setup/tick/
//     shutdown calls each sub-harness's setup/tick/shutdown in order
//     with cross-world connect() steps between them.
//
// User code can pass a `manta::Harness*` around — useful for plugin
// architectures, test rigs that swap harnesses at runtime, or generic
// multi-Target binaries that need to drive several Harnesses in
// lockstep.
//
// The free functions (`manta_gen::<name>::setup()` etc.) are still the
// preferred call surface for hot-path code: they're inlinable and
// non-virtual. The Harness class is for cases where polymorphism is
// genuinely useful.
class Harness {
public:
    virtual ~Harness() = default;

    // One-time initialization. Called once before the first tick().
    virtual void setup() = 0;

    // One simulation step. The pacer (in `<target>_main.cpp` or in the
    // user's main when in library workflow) calls this in a loop.
    virtual void tick() = 0;

    // Tear down any state that can't survive static destruction —
    // notably the Zenoh session, whose Tokio-based runtime panics if
    // destroyed at process exit. Call before main() returns.
    virtual void shutdown() = 0;
};

} // namespace manta
