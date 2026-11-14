import importlib.util
from pathlib import Path

import pytest

from grpo_gsm8k.audit import AUDIT_COLUMNS, grader_agreement

APP_PATH = Path(__file__).resolve().parents[1] / "tools" / "handcheck_app.py"
_spec = importlib.util.spec_from_file_location("handcheck_app", APP_PATH)
app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app)  # safe: Streamlit is imported only inside main()


def make_sheet(path: Path) -> Path:
    rows = [
        {"problem_id": "p1", "question": "q1", "gold_answer": "3", "completion": "a\n  b", "human_correct": "1", "grader_lenient": "1"},
        {"problem_id": "p2", "question": "q2, with comma", "gold_answer": "4", "completion": 'say "4"', "human_correct": "", "grader_lenient": "0"},
        {"problem_id": "p3", "question": "q3", "gold_answer": "5", "completion": "", "human_correct": "", "grader_lenient": "1"},
    ]
    app.save_sheet_atomically(rows, path)
    return path


def test_round_trip_keeps_columns_order_and_empty_cells(tmp_path):
    path = make_sheet(tmp_path / "sheet.csv")
    rows = app.load_sheet(path)
    app.save_sheet_atomically(rows, path)
    assert path.read_text().splitlines()[0] == ",".join(AUDIT_COLUMNS)
    assert app.load_sheet(path) == rows
    assert [row["problem_id"] for row in rows] == ["p1", "p2", "p3"]
    assert rows[2]["completion"] == "" and rows[1]["human_correct"] == ""
    assert rows[0]["completion"] == "a\n  b"
    assert list(tmp_path.iterdir()) == [path]  # no temp file left behind


def test_first_unjudged_index_resumes(tmp_path):
    rows = app.load_sheet(make_sheet(tmp_path / "sheet.csv"))
    assert app.first_unjudged_index(rows) == 1
    app.set_judgement(rows, 1, "0")
    app.set_judgement(rows, 2, "1")
    assert app.first_unjudged_index(rows) == 2  # all judged: last row
    assert app.count_judged(rows) == 3


def test_set_judgement_rejects_other_values(tmp_path):
    rows = app.load_sheet(make_sheet(tmp_path / "sheet.csv"))
    with pytest.raises(ValueError):
        app.set_judgement(rows, 0, "yes")


def test_judging_then_agreement(tmp_path):
    path = make_sheet(tmp_path / "sheet.csv")
    rows = app.load_sheet(path)
    with pytest.raises(ValueError):
        grader_agreement(rows)
    app.set_judgement(rows, 1, "0")
    app.set_judgement(rows, 2, "0")
    app.save_sheet_atomically(rows, path)
    rate, disagreements = grader_agreement(app.load_sheet(path))
    assert rate == pytest.approx(2 / 3)
    assert [row["problem_id"] for row in disagreements] == ["p3"]


def test_app_smoke_hides_grader_and_saves(tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    sheet = make_sheet(tmp_path / "sheet.csv")
    at = AppTest.from_file(str(APP_PATH))
    at.session_state[app.SHEET_KEY] = str(sheet)
    at.run()
    assert not at.exception
    shown = [e.value for kind in (at.markdown, at.code, at.caption, at.subheader, at.title) for e in kind]
    assert not any("lenient" in str(text).lower() for text in shown)
    reveal = next(b for b in at.button if b.label == "Reveal agreement")
    assert reveal.disabled
    next(b for b in at.button if b.label == "Wrong (0)").click().run()
    assert app.load_sheet(sheet)[1]["human_correct"] == "0"
    next(b for b in at.button if b.label == "Correct (1)").click().run()
    assert app.count_judged(app.load_sheet(sheet)) == 3
    reveal = next(b for b in at.button if b.label == "Reveal agreement")
    assert not reveal.disabled
    reveal.click().run()
    assert not at.exception
    assert len(at.expander) == 0  # human and grader agree on every row
