"""The `[1m]` context suffix on a model that resolves through the catalog.

First-party ids stripped it before lookup, so `claude-opus-5-5[1m]` always
resolved. A catalog id did not: the remainder `[1m]` passed the separator test
in `_best_prefix` and then its `1` tripped the guard that rejects a new
generation, so `claude-sonnet-4-5[1m]` was an unknown model and its turns were
dropped from every total.
"""

from __future__ import annotations

import pytest

from adder.pricing import registry


@pytest.mark.parametrize("model", ["claude-sonnet-4-5", "claude-opus-4-5", "claude-sonnet-4"])
def test_the_suffix_resolves_to_the_same_model(model):
    assert registry.resolve(f"{model}[1m]").id == registry.resolve(model).id


def test_a_digit_after_a_dash_is_still_a_new_generation():
    assert not registry.is_known("claude-opus-6")
