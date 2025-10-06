import subprocess
from pathlib import Path


def test_status_script_exits_zero():
    root = Path(__file__).resolve().parents[1]
    script = root / "tools" / "status.sh"
    assert script.exists(), "tools/status.sh is missing"
    proc = subprocess.run(["bash", str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # We expect success even if tools are missing, since the script skips gracefully
    assert proc.returncode == 0, f"status.sh failed:\n{proc.stdout.decode()}"

