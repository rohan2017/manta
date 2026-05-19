"""manta_next — Python-first, CasADi-backed manta IR.

Authoring API at the top level:

    from manta_next import ir
    from manta_next.ir.frames import SceneFrame, CraftFrame

    with ir.Graph() as g:
        p = ir.Vec3[SceneFrame].input("p")
        v = ir.Vec3[CraftFrame].input("v")
        q = ir.Quat[SceneFrame, CraftFrame].input("q")
        dt = ir.Scalar.input("dt")

        p_next = p + q.apply(v) * dt
        g.output(p_next, "p_next")

    fn = g.compile()              # casadi.Function wrapped for kwarg I/O
    print(fn(p=[0,0,5], v=[1,0,0], q=[1,0,0,0], dt=0.001)["p_next"])

Nothing executes eagerly. User code always builds a graph; to run it, you
compile to a CasADi Function (= the implicit "Python backend") or hand
the graph to a different backend (C++/Eigen, C/embedded).
"""

from . import ir

__all__ = ["ir"]
