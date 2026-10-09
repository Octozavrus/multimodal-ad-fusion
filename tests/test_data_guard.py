import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "check_no_data", Path(__file__).resolve().parents[1] / "scripts" / "check_no_data.py"
)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


S = "_S_"  # IDs are assembled at runtime so this file passes the guard itself


@pytest.mark.parametrize(
    "text,blocked",
    [
        (f"subject 941{S}1194 was excluded", True),
        (f"path/ADNI/023{S}0031/MPRAGE", True),
        ("synthetic 000_S_0001 is allowed", False),
        ("regex \\d{3}_S_\\d{4}", False),
        ("1234_S_56789", False),
    ],
)
def test_real_id_pattern(text, blocked):
    assert bool(guard.REAL_ID.search(text)) is blocked


def test_repository_is_clean():
    assert guard.main() == 0
