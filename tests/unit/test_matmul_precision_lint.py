"""Every JAX matmul in relax's EM code passes an explicit precision.

JAX's default matmul precision lets XLA run float32 and complex64 products as TF32 on A100/H100
(about 3e-4 relative error) and in float32 on P100, so an unannotated product would make results
depend on the GPU generation. PPCA chooses its precision on purpose (fp32 | tf32) and is exempt.
``relax.helpers.dtype_policy.use_float32_matmuls`` is the EM commands' backstop for products this
scan cannot see (``@`` on JAX arrays).
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_EXEMPT = ("relax/ppca_initial_model/", "relax/ppca_refinement/", "relax/commands/ppca_initial_model.py")
_MATMULS = {
    ("jnp", "dot"),
    ("jnp", "matmul"),
    ("jnp", "einsum"),
    ("jnp", "tensordot"),
    ("jnp", "inner"),
    ("jnp", "vdot"),
    ("lax", "dot"),
    ("lax", "dot_general"),
    ("lax", "batch_matmul"),
}


def _callee(node: ast.Call) -> tuple[str, str] | None:
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    owner = func.value
    if isinstance(owner, ast.Name):
        return owner.id, func.attr
    if isinstance(owner, ast.Attribute) and isinstance(owner.value, ast.Name):
        # jax.numpy.dot, jax.lax.dot_general
        if owner.value.id == "jax" and owner.attr in {"numpy", "lax"}:
            return ("jnp" if owner.attr == "numpy" else "lax"), func.attr
    return None


def _unannotated_matmuls():
    found = []
    for path in sorted((REPO / "relax").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith(_EXEMPT):
            continue
        for node in ast.walk(ast.parse(path.read_text(), filename=rel)):
            if isinstance(node, ast.Call) and _callee(node) in _MATMULS:
                if not any(keyword.arg == "precision" for keyword in node.keywords):
                    found.append(f"{rel}:{node.lineno}")
    return found


def test_every_em_jax_matmul_passes_an_explicit_precision():
    assert _unannotated_matmuls() == []


def test_the_scan_sees_an_unannotated_matmul(tmp_path):
    tree = ast.parse("import jax.numpy as jnp\njnp.einsum('ij,jk->ik', a, b)\njax.lax.dot_general(a, b, dims)\n")
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert {_callee(call) for call in calls} >= {("jnp", "einsum"), ("lax", "dot_general")}


def test_em_commands_default_to_float32_matmuls():
    import jax

    from relax.helpers.dtype_policy import use_float32_matmuls

    previous = jax.config.jax_default_matmul_precision
    try:
        use_float32_matmuls()
        assert jax.config.jax_default_matmul_precision == "highest"
    finally:
        jax.config.update("jax_default_matmul_precision", previous)
