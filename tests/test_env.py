"""Stage A environment criteria E-1..E-7 (docs/experiment_protocol.md section 7.1)."""
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / "reports" / "logs" / "env_evidence"
pytestmark = [pytest.mark.stage("A"), pytest.mark.kind("DET")]


@pytest.mark.criterion("E-1")
def test_E1_global_environment_unchanged_and_venv_used():
    assert ".venv" in sys.prefix, f"tests must run inside the project venv, got {sys.prefix}"
    assert (EVID / "E1_status.txt").read_text().strip() == "PASS"
    assert (EVID / "global_pip_diff.txt").read_text() == ""


@pytest.mark.criterion("E-2")
def test_E2_torch_cpu_build():
    import torch
    assert torch.__version__ == "2.14.1+cpu"
    assert torch.version.cuda is None


@pytest.mark.criterion("E-3")
def test_E3_pandas_below_3():
    import pandas
    assert int(pandas.__version__.split(".")[0]) < 3


@pytest.mark.criterion("E-4")
@pytest.mark.parametrize("mod", ["numpy", "scipy", "pandas", "sklearn", "torch", "lifelines", "sksurv",
                                 "yaml", "matplotlib", "pytest", "hypothesis", "rdata"])
def test_E4_full_dependency_set_imports(mod):
    m = importlib.import_module(mod)
    print(mod, getattr(m, "__version__", "?"))


@pytest.mark.criterion("E-5")
def test_E5_lockfile_reproduces_exactly():
    assert (EVID / "E5_status.txt").read_text().strip() == "PASS"
    assert (EVID / "lockcheck_diff.txt").read_text() == ""
    assert (ROOT / "requirements-lock.txt").read_text() == (EVID / "lockcheck_freeze.txt").read_text()


@pytest.mark.criterion("E-6")
def test_E6_pip_check_clean():
    out = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


@pytest.mark.criterion("E-7")
def test_E7_pytest_collects_and_runs():
    assert True
