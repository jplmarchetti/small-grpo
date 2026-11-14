"""Streamlit page to fill the `human_correct` column of a hand-check sheet.

Run from the repo root:
    uv run --with streamlit streamlit run tools/handcheck_app.py -- --sheet audits/preflight_B0_fewshot.csv

The grader's verdict (`grader_lenient`) is never shown while judging, so the human
judgement stays blind. Every click saves the sheet atomically (temp file, then rename).
"""

import argparse
import csv
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from grpo_gsm8k.audit import AUDIT_COLUMNS, NO, YES, grader_agreement, read_audit_sheet

SHEET_KEY = "sheet_path"


# ---------- pure helpers (testable without Streamlit) ----------


def load_sheet(path: Path) -> list[dict]:
    return read_audit_sheet(path)


def is_judged(row: dict) -> bool:
    return row["human_correct"].strip() in (YES, NO)


def count_judged(rows: Sequence[dict]) -> int:
    return sum(is_judged(row) for row in rows)


def first_unjudged_index(rows: Sequence[dict]) -> int:
    """Index of the first row without a judgement; the last row if all are judged."""
    for index, row in enumerate(rows):
        if not is_judged(row):
            return index
    return max(len(rows) - 1, 0)


def set_judgement(rows: list[dict], index: int, value: str) -> None:
    if value not in (YES, NO):
        raise ValueError(f"judgement must be {YES!r} or {NO!r}, got {value!r}")
    rows[index]["human_correct"] = value


def save_sheet_atomically(rows: Sequence[dict], path: Path) -> None:
    """Write to a temp file in the same folder, then rename over the sheet."""
    descriptor, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


# ---------- Streamlit page ----------


def sheet_path_from_args(argv: Sequence[str]) -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheet", required=True, type=Path)
    args, _ = parser.parse_known_args(argv)
    return args.sheet


def render_row(st, row: dict) -> None:
    st.subheader(f"Problem {row['problem_id']}")
    st.markdown("**Question**")
    st.write(row["question"])
    st.markdown(f"**Gold answer:** `{row['gold_answer']}`")
    st.markdown("**Completion**")
    st.code(row["completion"], language=None)


def render_navigation(st, rows: Sequence[dict]) -> None:
    index = st.session_state.index
    previous_col, next_col = st.columns(2)
    if previous_col.button("Previous", disabled=index == 0, use_container_width=True):
        st.session_state.index = index - 1
        st.rerun()
    if next_col.button("Next", disabled=index >= len(rows) - 1, use_container_width=True):
        st.session_state.index = index + 1
        st.rerun()


def render_judgement_buttons(st, rows: list[dict], path: Path) -> None:
    index = st.session_state.index
    current = rows[index]["human_correct"].strip()
    if current:
        st.caption(f"Your current judgement: {'Correct' if current == YES else 'Wrong'}")
    correct_col, wrong_col = st.columns(2)
    for column, label, value in ((correct_col, "Correct (1)", YES), (wrong_col, "Wrong (0)", NO)):
        if column.button(label, type="primary", use_container_width=True, key=f"judge_{value}"):
            set_judgement(rows, index, value)
            save_sheet_atomically(rows, path)
            st.session_state.index = min(index + 1, len(rows) - 1)
            st.rerun()


def render_reveal(st, rows: Sequence[dict]) -> None:
    all_judged = count_judged(rows) == len(rows)
    if not st.button("Reveal agreement", disabled=not all_judged):
        if not all_judged:
            st.caption("Judge every row to unlock the reveal.")
        return
    rate, disagreements = grader_agreement(rows)
    st.metric("Agreement with lenient grader", f"{rate:.1%}")
    st.write(f"{len(disagreements)} disagreement(s)")
    for row in disagreements:
        title = (
            f"{row['problem_id']} | gold {row['gold_answer']} | "
            f"grader {row['grader_lenient']} | human {row['human_correct']}"
        )
        with st.expander(title):
            st.code(row["completion"], language=None)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Hand-check", layout="centered")
    path = Path(st.session_state.get(SHEET_KEY) or sheet_path_from_args(sys.argv[1:]))
    if "rows" not in st.session_state:
        st.session_state.rows = load_sheet(path)
        st.session_state.index = first_unjudged_index(st.session_state.rows)
    rows = st.session_state.rows
    if not rows:
        st.error(f"No rows in {path}")
        return

    st.title("Hand-check: did the model get the right final answer?")
    st.progress(count_judged(rows) / len(rows), text=f"{count_judged(rows)} / {len(rows)} judged")
    st.caption(f"Row {st.session_state.index + 1} of {len(rows)} | {path}")
    render_row(st, rows[st.session_state.index])
    render_judgement_buttons(st, rows, path)
    render_navigation(st, rows)
    st.divider()
    render_reveal(st, rows)


if __name__ == "__main__":
    main()
