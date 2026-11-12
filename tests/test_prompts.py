from grpo_gsm8k.data import Problem
from grpo_gsm8k.graders import grade_completion
from grpo_gsm8k.prompts import STOP_STRING, build_prompt, format_exemplar

# METHODOLOGY §4, copied character by character.
EXPECTED_ZERO_SHOT = """A conversation between User and Assistant. The User asks a math question and the
Assistant solves it step by step, then writes the final answer on its own last
line in the form "FINAL: <number>".

User: How many legs do 3 cats have?
Assistant:"""

EXEMPLAR = Problem(
    problem_id="train-01123",
    question="David found $12 on the street. He then gave it to his friend Evan who has $1 and needed to buy a watch worth $20. How much does Evan still need?",
    gold_answer="7",
    solution=(
        "The total amount Evan has is $1 + $12 = $<<1+12=13>>13.\n"
        "So Evan still needs $20 - $13 = $<<20-13=7>>7.\n"
        "#### 7"
    ),
)


def test_zero_shot_prompt_matches_methodology():
    assert build_prompt("How many legs do 3 cats have?") == EXPECTED_ZERO_SHOT


def test_question_whitespace_is_stripped():
    assert build_prompt("  How many legs do 3 cats have?\n") == EXPECTED_ZERO_SHOT


def test_exemplar_is_rewritten_into_the_final_format():
    assert format_exemplar(EXEMPLAR) == (
        "The total amount Evan has is $1 + $12 = $13.\n"
        "So Evan still needs $20 - $13 = $7.\n"
        "FINAL: 7"
    )


def test_rewritten_exemplar_gets_full_marks_from_the_grader():
    grade = grade_completion(format_exemplar(EXEMPLAR), EXEMPLAR.gold_answer)
    assert (grade.format_ok, grade.strict_correct, grade.lenient_correct) == (True, True, True)


def test_fewshot_prompt_is_zero_shot_with_examples_inserted():
    zero_shot = build_prompt("How many legs do 3 cats have?")
    few_shot = build_prompt("How many legs do 3 cats have?", exemplars=[EXEMPLAR, EXEMPLAR])
    preamble, question_turn = zero_shot.rsplit("\n\n", 1)
    assert few_shot.startswith(preamble + "\n\nUser: David found")
    assert few_shot.endswith("\n\n" + question_turn)
    assert few_shot.count("FINAL: 7") == 2
    assert "<<" not in few_shot and "####" not in few_shot


def test_prompts_end_with_assistant_and_the_stop_string_marks_a_new_turn():
    prompt = build_prompt("q", exemplars=[EXEMPLAR])
    assert prompt.endswith("\nAssistant:")
    # every User turn after the preamble begins with the stop string the model is cut at
    assert prompt.count(STOP_STRING) == 2
