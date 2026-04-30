"""Emit <name>.hpp + <name>.cpp — the typed Craft subclass.

Two modes:
  * Non-templated (default): emits `class FooCraft : public manta::Craft`
    with a separate .cpp constructor body. Existing path.
  * Scalar-templated (when `craft.scalar_templated == True`): emits
    `template <class Scalar = manta::Real> class FooCraftT : public
    manta::CraftT<Scalar>` as a header-only class, plus a
    `using FooCraft = FooCraftT<manta::Real>` alias. Required for use with
    `manta::estimation::CraftEKF`. All parts must have `cpp_class_template`
    set; codegen raises otherwise.
"""

from __future__ import annotations

from ..core import Craft
from ._util import GENERATED_BANNER, CPP_INCLUDE_GUARD, class_name_for_craft


def _validate_templated(craft: Craft) -> None:
    bad = [p.name for p in craft.all_parts() if not p.cpp_class_template]
    if bad:
        raise ValueError(
            f"Craft {craft.name!r}: scalar_templated=True requires every "
            f"part to declare `cpp_class_template`. Parts missing it: {bad}. "
            "Either template those parts (preferred) or set "
            "scalar_templated=False on this craft."
        )


def emit_craft_hpp(craft: Craft) -> str:
    if craft.scalar_templated:
        return _emit_craft_hpp_templated(craft)
    return _emit_craft_hpp_concrete(craft)


def emit_craft_cpp(craft: Craft) -> str:
    if craft.scalar_templated:
        # Header-only when templated — the .cpp is just a stub for build symmetry.
        cls = class_name_for_craft(craft.name)
        return (
            f"{GENERATED_BANNER}\n"
            f"// {cls} is templated and emitted entirely in {craft.name}.hpp.\n"
            f'#include "{craft.name}.hpp"\n'
        )
    return _emit_craft_cpp_concrete(craft)


# ---------------------------------------------------------------------------
# Non-templated emission (existing behavior)

def _emit_craft_hpp_concrete(craft: Craft) -> str:
    cls = class_name_for_craft(craft.name)
    parts = list(craft.all_parts())

    seen: set[str] = set()
    part_headers: list[str] = []
    for p in parts:
        if p.cpp_header and p.cpp_header not in seen:
            seen.add(p.cpp_header)
            part_headers.append(p.cpp_header)

    accessors = []
    for p in parts:
        accessors.append(f"    {p.cpp_class}& {p.name}() {{ return *{p.name}_; }}")
        accessors.append(f"    const {p.cpp_class}& {p.name}() const {{ return *{p.name}_; }}")

    members = [f"    {p.cpp_class}* {p.name}_ = nullptr;" for p in parts]

    lines: list[str] = [
        GENERATED_BANNER, CPP_INCLUDE_GUARD, "",
        '#include "manta/core/craft.hpp"',
    ]
    lines += [f'#include "{h}"' for h in part_headers]
    lines += [
        "",
        f"class {cls} : public manta::Craft {{",
        "public:",
        f"    {cls}();",
        "",
    ]
    lines += accessors
    lines += ["", "private:"]
    lines += members
    lines += ["};", ""]
    return "\n".join(lines)


def _emit_craft_cpp_concrete(craft: Craft) -> str:
    cls = class_name_for_craft(craft.name)
    lines: list[str] = [
        GENERATED_BANNER,
        f'#include "{craft.name}.hpp"',
        "",
        f"{cls}::{cls}()",
        f'    : manta::Craft("{craft.name}") {{',
    ]

    def emit_subtree(parent_handle: str, part) -> None:
        ctor_args = part.emit_constructor_args()
        lines.append(
            f"    {part.name}_ = &{parent_handle}.add<{part.cpp_class}>({ctor_args});"
        )
        for stmt in part.emit_post_construction():
            lines.append(f"    {stmt}")
        for child in part._children:
            emit_subtree(f"(*{part.name}_)", child)

    for p in craft.root.children:
        emit_subtree("this->root()", p)

    lines.append("    this->root().compute_params();")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scalar-templated emission

def _emit_craft_hpp_templated(craft: Craft) -> str:
    _validate_templated(craft)

    cls_t = class_name_for_craft(craft.name) + "T"
    cls   = class_name_for_craft(craft.name)
    parts = list(craft.all_parts())

    seen: set[str] = set()
    part_headers: list[str] = []
    for p in parts:
        if p.cpp_header and p.cpp_header not in seen:
            seen.add(p.cpp_header)
            part_headers.append(p.cpp_header)

    lines: list[str] = [
        GENERATED_BANNER, CPP_INCLUDE_GUARD, "",
        '#include "manta/core/craft.hpp"',
    ]
    lines += [f'#include "{h}"' for h in part_headers]
    lines += [
        "",
        f"template <class Scalar = manta::Real>",
        f"class {cls_t} : public manta::CraftT<Scalar> {{",
        "public:",
        f"    {cls_t}() : manta::CraftT<Scalar>(\"{craft.name}\") {{",
    ]

    def emit_subtree(parent_handle: str, part) -> None:
        ctor_args = part.emit_constructor_args(scalar="Scalar")
        ct        = part.cpp_class_template_instantiation("Scalar")
        # `template add<Foo<Scalar>>` requires the `template` disambiguator
        # because the inheritance is dependent on the template parameter.
        lines.append(
            f"        {part.name}_ = &{parent_handle}.template add<"
            f"{ct}>({ctor_args});"
        )
        for stmt in part.emit_post_construction(scalar="Scalar"):
            lines.append(f"        {stmt}")
        for child in part._children:
            emit_subtree(f"(*{part.name}_)", child)

    for p in craft.root.children:
        emit_subtree("this->root()", p)

    lines.append("        this->root().compute_params();")
    lines.append("    }")
    lines.append("")

    # Accessors.
    for p in parts:
        ct = p.cpp_class_template_instantiation("Scalar")
        lines.append(f"    {ct}& {p.name}() {{ return *{p.name}_; }}")
        lines.append(f"    const {ct}& {p.name}() const {{ return *{p.name}_; }}")

    lines += ["", "private:"]
    for p in parts:
        ct = p.cpp_class_template_instantiation("Scalar")
        lines.append(f"    {ct}* {p.name}_ = nullptr;")
    lines += ["};", "", f"using {cls} = {cls_t}<manta::Real>;", ""]
    return "\n".join(lines)
