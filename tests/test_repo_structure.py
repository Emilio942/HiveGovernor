from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_required_directories_exist():
    required = [
        ROOT / "swarm_moi",
        ROOT / "tests",
        ROOT / "examples",
        ROOT / "plans",
        ROOT / "tools",
    ]
    for p in required:
        assert p.exists() and p.is_dir(), f"Missing directory: {p}"


def test_package_init_exists_and_imports():
    pkg_init = ROOT / "swarm_moi" / "__init__.py"
    assert pkg_init.exists(), "Missing swarm_moi/__init__.py"
    # Import should succeed
    __import__("swarm_moi")


def test_dependency_manifest_present():
    # Either pyproject.toml or requirements.txt should exist
    pyproject = ROOT / "pyproject.toml"
    requirements = ROOT / "requirements.txt"
    assert pyproject.exists() or requirements.exists(), (
        "Expected either pyproject.toml or requirements.txt"
    )

