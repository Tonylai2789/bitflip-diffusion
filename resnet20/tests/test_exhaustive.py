"""Resume support for the exhaustive run: faults.csv is cut back to whole elements (16 rows each)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from resnet20_fault import all_elements, build_model, truncate_to_elements  # noqa: E402


def _write(path, n_rows, tail=b""):
    path.write_bytes(b"header\n" + b"".join(b"row%d\n" % i for i in range(n_rows)) + tail)


def test_partial_element_and_partial_line_are_dropped(tmp_path):
    p = tmp_path / "faults.csv"
    _write(p, 35, tail=b"row35,half-writ")  # 2 whole elements + 3 rows + a partial line
    assert truncate_to_elements(p, 16) == 2
    assert p.read_bytes() == b"header\n" + b"".join(b"row%d\n" % i for i in range(32))


def test_whole_elements_untouched(tmp_path):
    p = tmp_path / "faults.csv"
    _write(p, 32)
    before = p.read_bytes()
    assert truncate_to_elements(p, 16) == 2
    assert p.read_bytes() == before
    _write(p, 0)
    assert truncate_to_elements(p, 16) == 0


def test_all_elements_covers_every_weight_once():
    m = build_model(1)
    elems = list(all_elements(m))
    assert len(elems) == 268_048 == len(set(elems))
    assert elems[0] == ("stage1", "conv1.weight", 0)
    assert elems[-1] == ("fc", "fc.weight", 639)
