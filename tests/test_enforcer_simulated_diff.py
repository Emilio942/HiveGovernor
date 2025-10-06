import importlib.util
import sys
from pathlib import Path


def _load_enforcer():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "tools" / "enforce_guards.py"
    spec = importlib.util.spec_from_file_location("enforce_guards", str(mod_path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Ensure module is visible to dataclasses/type evaluation under future annotations
    sys.modules[spec.name] = mod  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_enforcer_allows_changes_within_guards():
    enforcer = _load_enforcer()
    diff = """
diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1,4 +1,6 @@
 line1
 >>> BEGIN:AI_EDIT
 context
 >>> END:AI_EDIT
+>>> BEGIN:AI_EDIT
+added_inside
+>>> END:AI_EDIT
""".lstrip()
    res = enforcer.check_diff_text(diff)
    assert res.ok, f"Expected OK, got errors: {res.errors}"


def test_enforcer_rejects_outside_guard():
    enforcer = _load_enforcer()
    diff = """
diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1,4 +1,5 @@
 line1
 >>> BEGIN:AI_EDIT
 context
 >>> END:AI_EDIT
+added_outside
""".lstrip()
    res = enforcer.check_diff_text(diff)
    assert not res.ok
    assert any("outside guard" in e for e in res.errors)


def test_enforcer_limits_files_and_lines():
    enforcer = _load_enforcer()
    # Build a diff with 4 files (exceeds default 3) and >200 lines inside guards
    parts = []
    for i in range(4):
        parts.append(
            f"""
diff --git a/f{i}.py b/f{i}.py
--- a/f{i}.py
+++ b/f{i}.py
@@ -1,1 +1,205 @@
 >>> BEGIN:AI_EDIT
""".lstrip()
        )
        # 205 additions
        for _ in range(205):
            parts.append("+x = 1\n")
        parts.append(">>> END:AI_EDIT\n")
    diff = "".join(parts)

    res = enforcer.check_diff_text(diff)
    assert not res.ok
    # Should trip both limits
    assert any("Too many files" in e for e in res.errors)
    assert any("Too many lines" in e for e in res.errors)
