"""Stop-trimming with a fake character tokenizer: token i is the i-th character."""

from grpo_gsm8k.config import load_config
from grpo_gsm8k.rollout import greedy_decoding, sampling_decoding, trim_at_stop

END = 0  # fake end-of-text token, decoded as nothing (special token)


def encode(text: str) -> list[int]:
    return [ord(character) for character in text]


def decode(ids: list[int]) -> str:
    return "".join(chr(i) for i in ids if i != END)


def test_stop_string_cuts_text_but_keeps_stop_tokens():
    ids, text, stopped = trim_at_stop(encode("so 18\nFINAL: 18\nUser: next question"), decode, {END})
    assert stopped
    assert text == "so 18\nFINAL: 18"
    assert decode(ids) == "so 18\nFINAL: 18\nUser:"  # trained on, judgment call 6
    assert "User:" not in text


def test_end_token_cuts_and_is_kept():
    ids, text, stopped = trim_at_stop(encode("FINAL: 3") + [END, END, END], decode, {END})
    assert stopped
    assert text == "FINAL: 3"
    assert ids == encode("FINAL: 3") + [END]


def test_padding_after_a_stop_string_is_dropped():
    """Rows that finished early are padded with end tokens; the stop string comes first."""
    ids, text, stopped = trim_at_stop(encode("ok\nUser:") + [END, END], decode, {END})
    assert stopped and text == "ok" and ids == encode("ok\nUser:")


def test_no_stop_means_truncated():
    ids, text, stopped = trim_at_stop(encode("still thinking about User"), decode, {END})
    assert not stopped
    assert text == "still thinking about User"
    assert ids == encode("still thinking about User")


def test_user_without_newline_is_not_a_stop():
    _, text, stopped = trim_at_stop(encode("the User: said 4"), decode, {END})
    assert not stopped and text == "the User: said 4"


def test_earliest_stop_wins():
    _, text, stopped = trim_at_stop(encode("a\nUser: b") + [END] + encode("c"), decode, {END})
    assert stopped and text == "a"


def test_empty_completion():
    ids, text, stopped = trim_at_stop([END], decode, {END})
    assert stopped and text == "" and ids == [END]


def test_decoding_settings_come_from_the_config():
    training = load_config().training
    sampling, greedy = sampling_decoding(training), greedy_decoding(training)
    assert (sampling.do_sample, sampling.temperature, sampling.top_p) == (True, 0.8, 0.95)
    assert not greedy.do_sample
    assert sampling.max_new_tokens == greedy.max_new_tokens == 512
