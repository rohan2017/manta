"""The "Build a Mako" compile server.

The browser (web/mako/) assembles a submarine *design* — spine modules,
mounted thrusters/sensors/control surfaces, a spawn pose — and POSTs it to
this package, which builds the manta `World` (a real spinning Earth with a
polar dive-site `Scene`), lowers the `Sim` to WebAssembly with
`TargetWasm`, and serves the cached bundle back. The
client/server contract is `web/mako/SPEC.md`.

    catalog.py   — the module-type registry (option schemas + part builders)
    builder.py   — design validation, World construction, trim, bundle emission
    __main__.py  — the dev HTTP server (`python -m server.mako`, port 8077)
"""
