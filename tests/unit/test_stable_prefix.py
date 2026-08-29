"""Contract tests for stable-prefix calculation over rolling hypotheses.

``is_stable`` on a partial transcript is a promise to the client, so the rules
that decide it are pinned exactly: agreement across the window, and never
committing half a word.
"""

from __future__ import annotations

import pytest

from app.audio.stable_prefix import (
    RollingStablePrefix,
    longest_common_prefix,
    stable_common_prefix,
)


class TestLongestCommonPrefix:
    def test_no_hypotheses_have_no_prefix(self) -> None:
        assert longest_common_prefix([]) == ""

    def test_a_single_hypothesis_is_entirely_its_own_prefix(self) -> None:
        assert longest_common_prefix(["namaste duniya"]) == "namaste duniya"

    def test_identical_hypotheses_agree_completely(self) -> None:
        assert longest_common_prefix(["abc", "abc", "abc"]) == "abc"

    def test_the_shared_character_run_is_returned(self) -> None:
        assert longest_common_prefix(["abcdef", "abcxyz"]) == "abc"

    def test_a_divergent_first_character_leaves_nothing(self) -> None:
        assert longest_common_prefix(["abc", "xbc"]) == ""

    def test_an_empty_hypothesis_collapses_the_prefix(self) -> None:
        assert longest_common_prefix(["abc", ""]) == ""

    def test_a_shorter_hypothesis_bounds_the_prefix(self) -> None:
        assert longest_common_prefix(["abcdef", "abc"]) == "abc"

    def test_comparison_is_by_character_not_by_word(self) -> None:
        assert longest_common_prefix(["hello world", "hello wide"]) == "hello w"

    def test_an_iterator_is_consumed_correctly(self) -> None:
        assert longest_common_prefix(iter(["prefix-a", "prefix-b"])) == "prefix-"


class TestStableCommonPrefix:
    def test_no_hypotheses_are_never_stable(self) -> None:
        assert stable_common_prefix([]) == ""

    def test_unanimous_hypotheses_are_fully_stable(self) -> None:
        hypotheses = ["mera naam", "mera naam"]
        assert stable_common_prefix(hypotheses) == "mera naam"

    def test_a_single_hypothesis_is_fully_stable(self) -> None:
        assert stable_common_prefix(["mera naa"]) == "mera naa"

    def test_an_incomplete_final_word_is_withheld(self) -> None:
        assert stable_common_prefix(["mera naam", "mera naaz"]) == "mera"

    def test_agreement_shorter_than_one_word_is_withheld(self) -> None:
        assert stable_common_prefix(["mera", "mero"]) == ""

    def test_trailing_whitespace_is_trimmed_from_a_word_boundary(self) -> None:
        assert stable_common_prefix(["ek do ", "ek do teen"]) == "ek do"

    def test_the_boundary_rule_can_be_disabled(self) -> None:
        assert stable_common_prefix(["mera naam", "mera naaz"], complete_words=False) == "mera naa"

    def test_disabling_the_boundary_rule_keeps_raw_whitespace(self) -> None:
        assert stable_common_prefix(["ek do ", "ek do teen"], complete_words=False) == "ek do "

    def test_multiple_agreed_words_are_all_stable(self) -> None:
        assert stable_common_prefix(["ek do teen char", "ek do teen chaar"]) == "ek do teen"

    def test_one_divergent_hypothesis_is_enough_to_withhold_text(self) -> None:
        assert stable_common_prefix(["ek do teen", "ek do teen", "ek nau"]) == "ek"

    def test_an_empty_hypothesis_makes_nothing_stable(self) -> None:
        assert stable_common_prefix(["ek do teen", ""]) == ""

    def test_stability_is_a_pure_function_of_the_hypotheses(self) -> None:
        hypotheses = ("ek do teen", "ek do teeen")
        assert stable_common_prefix(hypotheses) == stable_common_prefix(hypotheses)


class TestRollingStablePrefix:
    def test_a_window_needs_at_least_two_hypotheses(self) -> None:
        for window_size in (-1, 0, 1):
            with pytest.raises(ValueError, match="at least two"):
                RollingStablePrefix(window_size)

    def test_the_first_hypothesis_alone_is_never_stable(self) -> None:
        assert RollingStablePrefix().add("ek do teen") == ""

    def test_the_stable_prefix_trails_the_newest_hypothesis_by_one_word(self) -> None:
        """The last shared word may still be growing, so it is not committed."""

        rolling = RollingStablePrefix()
        rolling.add("ek do")
        assert rolling.add("ek do teen") == "ek"

    def test_growing_agreement_extends_the_stable_prefix(self) -> None:
        rolling = RollingStablePrefix(window_size=2)
        assert rolling.add("ek") == ""
        assert rolling.add("ek do") == ""
        assert rolling.add("ek do teen") == "ek"
        assert rolling.add("ek do teen char") == "ek do"
        assert rolling.add("ek do teen char paanch") == "ek do teen"

    def test_a_growing_transcript_never_retracts_stable_text(self) -> None:
        """The property the client depends on: committed text stays committed."""

        rolling = RollingStablePrefix(window_size=3)
        hypotheses = [
            "ek",
            "ek do",
            "ek do teen",
            "ek do teen char",
            "ek do teen char paanch",
            "ek do teen char paanch chhah",
        ]
        stable_history: list[str] = []
        for hypothesis in hypotheses:
            stable = rolling.add(hypothesis)
            if stable_history:
                assert stable.startswith(stable_history[-1])
            assert hypothesis.startswith(stable)
            stable_history.append(stable)
        assert stable_history == ["", "", "", "ek", "ek do", "ek do teen"]

    def test_a_revised_hypothesis_can_retract_the_computed_prefix(self) -> None:
        """A rewrite shrinks agreement, so callers must not re-emit stability."""

        rolling = RollingStablePrefix(window_size=2)
        assert rolling.add("ek do teen") == ""
        assert rolling.add("ek do teen char") == "ek do"
        assert rolling.add("ek nau") == "ek"

    def test_the_window_is_bounded_and_forgets_old_hypotheses(self) -> None:
        rolling = RollingStablePrefix(window_size=2)
        rolling.add("alpha beta")
        rolling.add("gamma delta")
        assert rolling.add("gamma delta epsilon") == "gamma"

    def test_a_wider_window_demands_agreement_for_longer(self) -> None:
        rolling = RollingStablePrefix(window_size=3)
        rolling.add("alpha beta")
        rolling.add("gamma delta")
        assert rolling.add("gamma delta epsilon") == ""

    def test_reset_starts_a_new_utterance_from_nothing(self) -> None:
        rolling = RollingStablePrefix(window_size=2)
        rolling.add("ek do")
        rolling.add("ek do teen")
        rolling.reset()
        assert rolling.add("naya vaakya") == ""

    def test_repeated_identical_hypotheses_commit_the_whole_text(self) -> None:
        rolling = RollingStablePrefix(window_size=2)
        rolling.add("poora vaakya")
        assert rolling.add("poora vaakya") == "poora vaakya"
