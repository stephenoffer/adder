"""Reading a usage record written by something other than Claude Code.

The failure this file exists to prevent is one number: providers disagree about
whether the cached prefix is *inside* the input count. Anthropic reports the
three counts disjointly; OpenAI and Google report a total with the cached part
included. Carrying an OpenAI record across verbatim counts the whole cached
prefix twice, and in an agent session the cached prefix is the bill.
"""

from __future__ import annotations

import json

import pytest

from adder.core import ingest
from adder.core.trace import iter_file, load_sessions


def _openai(prompt, cached, out=500, reasoning=0, mid="m1", model="gpt-5"):
    return {
        "model": model, "id": mid, "created": 1767225600,
        "usage": {
            "prompt_tokens": prompt, "completion_tokens": out,
            "prompt_tokens_details": {"cached_tokens": cached},
            "completion_tokens_details": {"reasoning_tokens": reasoning},
        },
    }


class TestSniff:
    @pytest.mark.parametrize("rec,expect", [
        ({"type": "assistant", "message": {"model": "claude-opus-5", "usage": {}}},
         ingest.CLAUDE_CODE),
        ({"type": "message", "model": "claude-opus-5", "usage": {"input_tokens": 1}},
         ingest.ANTHROPIC_API),
        ({"model": "gpt-5", "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
         ingest.OPENAI_CHAT),
        ({"model": "gpt-5", "usage": {"input_tokens": 1, "output_tokens": 1}},
         ingest.OPENAI_RESPONSES),
        ({"modelVersion": "gemini-3-flash", "usageMetadata": {"promptTokenCount": 1}},
         ingest.GEMINI),
        ({"attributes": {"gen_ai.usage.input_tokens": 1,
                         "gen_ai.request.model": "gpt-5"}}, ingest.OTEL),
    ])
    def test_each_shape_is_recognised(self, rec, expect):
        assert ingest.sniff(rec) == expect

    def test_a_record_with_no_usage_is_not_a_turn(self):
        assert ingest.sniff({"hello": "world"}) is None
        assert ingest.sniff({"type": "user", "content": "hi"}) is None

    def test_responses_and_messages_api_are_told_apart(self):
        """They use the same two field names for opposite conventions.

        `input_tokens` excludes the cache on Anthropic and includes it on
        OpenAI's Responses API. One spelling, a factor of two on the input
        side. `cache_creation_input_tokens` is the tell.
        """
        anthropic = {"model": "claude-opus-5",
                     "usage": {"input_tokens": 10, "output_tokens": 1,
                               "cache_creation_input_tokens": 5}}
        openai = {"model": "gpt-5", "usage": {"input_tokens": 10, "output_tokens": 1}}
        assert ingest.sniff(anthropic) == ingest.ANTHROPIC_API
        assert ingest.sniff(openai) == ingest.OPENAI_RESPONSES

    def test_sniffing_is_per_record_not_per_file(self):
        """A proxy log interleaves three providers in one file."""
        mixed = [_openai(100, 90),
                 {"modelVersion": "gemini-3-flash",
                  "usageMetadata": {"promptTokenCount": 5}}]
        assert [ingest.sniff(r) for r in mixed] == [ingest.OPENAI_CHAT, ingest.GEMINI]


class TestTheOverlapCorrection:
    def test_openai_cached_tokens_are_subtracted_from_the_prompt_total(self):
        """`prompt_tokens` INCLUDES `cached_tokens`. The context is 50,000,
        not 98,000."""
        t = ingest.turn_from(_openai(50_000, 48_000), session="s")
        assert t.uncached_in == 2_000
        assert t.cache_read == 48_000
        assert t.context == 50_000

    def test_gemini_cached_content_is_subtracted_too(self):
        rec = {"modelVersion": "gemini-3-flash", "responseId": "g1",
               "usageMetadata": {"promptTokenCount": 30_000,
                                 "cachedContentTokenCount": 28_000,
                                 "candidatesTokenCount": 500}}
        t = ingest.turn_from(rec, session="s")
        assert (t.uncached_in, t.cache_read, t.context) == (2_000, 28_000, 30_000)

    def test_anthropic_counts_are_disjoint_and_nothing_is_subtracted(self):
        rec = {"type": "message", "model": "claude-opus-5", "id": "a1",
               "usage": {"input_tokens": 2_000, "cache_read_input_tokens": 48_000,
                         "cache_creation_input_tokens": 1_000, "output_tokens": 800}}
        t = ingest.turn_from(rec, session="s")
        assert (t.uncached_in, t.cache_read, t.cache_write) == (2_000, 48_000, 1_000)
        assert t.context == 51_000

    def test_a_log_that_disagrees_with_itself_does_not_go_negative(self):
        """`cached_tokens` above `prompt_tokens` is impossible, and happens."""
        t = ingest.turn_from(_openai(1_000, 5_000), session="s")
        assert t.uncached_in == 0
        assert t.cache_read == 1_000

    def test_otel_subtracts_only_for_the_providers_that_overlap(self):
        """The OTel convention does not say, so the provider decides.

        Guessing uniformly is wrong for half the corpus either way.
        """
        oai = ingest.turn_from({"attributes": {
            "gen_ai.response.model": "gpt-5",
            "gen_ai.usage.input_tokens": 10_000,
            "gen_ai.usage.cached_input_tokens": 9_000,
            "gen_ai.usage.output_tokens": 200}}, session="s")
        ant = ingest.turn_from({"attributes": {
            "gen_ai.response.model": "claude-opus-5",
            "gen_ai.usage.input_tokens": 10_000,
            "gen_ai.usage.cache_read_input_tokens": 9_000,
            "gen_ai.usage.output_tokens": 200}}, session="s")
        assert (oai.uncached_in, oai.context) == (1_000, 10_000)
        assert (ant.uncached_in, ant.context) == (10_000, 19_000)


class TestNoInventedCacheWrites:
    def test_automatic_caching_records_no_write(self):
        """Inventing a `cache_write` to look like Anthropic would re-introduce
        a 1.25x premium nobody was charged."""
        t = ingest.turn_from(_openai(50_000, 48_000), session="s")
        assert t.cache_write == 0

    def test_and_so_the_priced_write_side_is_zero(self):
        t = ingest.turn_from(_openai(50_000, 48_000), session="s")
        assert t.cost() == pytest.approx(t.input_cost() + t.output_cost())


class TestReasoningTokens:
    def test_openai_reasoning_is_recorded_without_double_counting_output(self):
        """`completion_tokens` already includes reasoning."""
        t = ingest.turn_from(_openai(100, 0, out=800, reasoning=300), session="s")
        assert t.out == 800
        assert t.thinking == 300

    def test_gemini_thoughts_are_outside_the_candidate_count_and_are_added(self):
        """Gemini is the one provider that reports them separately. Leaving
        them out under-bills every reasoning turn."""
        rec = {"modelVersion": "gemini-3-flash",
               "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 500,
                                 "thoughtsTokenCount": 200}}
        t = ingest.turn_from(rec, session="s")
        assert t.out == 700
        assert t.thinking == 200


class TestTimestamps:
    def test_epoch_seconds_become_a_timestamp(self):
        """Without this every idle-gap and TTL report silently has nothing to
        measure on an OpenAI log."""
        t = ingest.turn_from(_openai(100, 0), session="s")
        assert t.when is not None
        assert t.when.year == 2026

    def test_a_missing_timestamp_is_not_an_error(self):
        rec = {"model": "gpt-5", "usage": {"prompt_tokens": 5, "completion_tokens": 1}}
        assert ingest.turn_from(rec, session="s").ts is None


class TestTtlLabelling:
    def test_an_automatic_provider_records_its_own_label(self):
        """Not the string "5m". A single non-selectable lifetime has to be
        distinguishable from a workload that chose the short TTL."""
        assert ingest.turn_from(_openai(100, 0), session="s").ttl == "auto"

    def test_anthropic_write_ttl_is_read_off_the_split(self):
        rec = {"type": "message", "model": "claude-opus-5",
               "usage": {"input_tokens": 1, "output_tokens": 1,
                         "cache_creation_input_tokens": 100,
                         "cache_creation": {"ephemeral_1h_input_tokens": 90,
                                            "ephemeral_5m_input_tokens": 10}}}
        assert ingest.turn_from(rec, session="s").ttl == "1h"


class TestFileShapes:
    def test_jsonl_array_and_single_object_all_parse(self, tmp_path):
        rows = [_openai(100, 50, mid="a"), _openai(200, 100, mid="b")]
        (tmp_path / "l.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        (tmp_path / "a.json").write_text(json.dumps(rows))
        (tmp_path / "o.json").write_text(json.dumps(rows[0]))
        assert len(list(ingest.iter_turns(tmp_path / "l.jsonl"))) == 2
        assert len(list(ingest.iter_turns(tmp_path / "a.json"))) == 2
        assert len(list(ingest.iter_turns(tmp_path / "o.json"))) == 1

    def test_a_malformed_file_degrades_to_nothing_rather_than_raising(self, tmp_path):
        f = tmp_path / "bad.jsonl"
        f.write_text("not json\n{also not\n")
        assert list(ingest.iter_turns(f)) == []

    def test_duplicate_ids_are_deduplicated_keeping_the_completed_stream(self,
                                                                        tmp_path):
        """The same hazard that inflated the Claude Code numbers 1.78x."""
        partial = _openai(1_000, 0, out=10, mid="same")
        full = _openai(1_000, 0, out=900, mid="same")
        f = tmp_path / "s.jsonl"
        f.write_text("\n".join(json.dumps(r) for r in (partial, full)))
        got = list(ingest.iter_turns(f))
        assert len(got) == 1
        assert got[0].out == 900

    def test_records_without_ids_are_never_merged(self, tmp_path):
        """Without an id there is no way to tell a duplicate from a second call."""
        rec = {"model": "gpt-5",
               "usage": {"prompt_tokens": 100, "completion_tokens": 5}}
        f = tmp_path / "s.jsonl"
        f.write_text("\n".join(json.dumps(rec) for _ in range(3)))
        assert len(list(ingest.iter_turns(f))) == 3


class TestUnknownModels:
    def test_an_unknown_model_is_tallied_not_silently_dropped(self, tmp_path):
        f = tmp_path / "s.jsonl"
        f.write_text(json.dumps(_openai(100, 0, model="frobnicator-7")))
        seen: dict[str, int] = {}
        got = list(ingest.iter_turns(f, unknown=seen))
        assert got == []
        assert seen == {"frobnicator-7": 1}

    def test_a_known_model_without_a_price_is_tallied_too(self, tmp_path):
        # The catalog lists models it has no price for. Admitting one only
        # moved the failure to `cost()`, which raised out of the report.
        from adder.pricing import registry

        unpriced = next(e.id for e in registry.catalog() if registry.is_known(e.id)
                        and not registry.is_priced(e.id))
        f = tmp_path / "s.jsonl"
        f.write_text(json.dumps(_openai(100, 0, model=unpriced)))
        seen: dict[str, int] = {}
        assert list(ingest.iter_turns(f, unknown=seen)) == []
        assert seen == {unpriced: 1}


class TestItPlugsIntoTheExistingReports:
    def test_iter_file_falls_back_to_the_adapters(self, tmp_path):
        """The wiring that makes every existing report work unchanged."""
        f = tmp_path / "s.jsonl"
        f.write_text("\n".join(json.dumps(_openai(10_000, 9_000, mid=f"m{i}"))
                               for i in range(5)))
        turns = list(iter_file(f))
        assert len(turns) == 5
        assert all(t.model == "gpt-5" for t in turns)

    def test_a_claude_code_file_still_takes_the_claude_code_path(self, tmp_path):
        """The native parser also does the per-content-block dedup the
        adapters have no view of, so it must keep winning."""
        rec = {"type": "assistant", "sessionId": "s1", "timestamp": "2026-08-01T00:00:00Z",
               "message": {"id": "x", "model": "claude-opus-5",
                           "usage": {"input_tokens": 10, "cache_read_input_tokens": 20,
                                     "output_tokens": 5}}}
        f = tmp_path / "s.jsonl"
        f.write_text("\n".join(json.dumps(rec) for _ in range(3)))
        assert len(list(iter_file(f))) == 1        # deduplicated by message id

    def test_load_sessions_groups_foreign_turns(self, tmp_path):
        d = tmp_path / "proj"
        d.mkdir()
        (d / "s.jsonl").write_text("\n".join(
            json.dumps(_openai(10_000 + i, 9_000, mid=f"m{i}")) for i in range(20)))
        sessions = load_sessions(tmp_path)
        assert sum(s.n_turns for s in sessions.values()) == 20
        assert sum(s.cost for s in sessions.values()) > 0


class TestDetectFormats:
    def test_it_reports_what_a_directory_actually_holds(self, tmp_path):
        """"1,200 records, all unrecognised" beats an empty report."""
        (tmp_path / "a.jsonl").write_text("\n".join(
            json.dumps(_openai(1, 0, mid=str(i))) for i in range(3)))
        (tmp_path / "b.jsonl").write_text(json.dumps({"nothing": "here"}))
        got = ingest.detect_formats(sorted(tmp_path.glob("*.jsonl")))
        assert got[ingest.OPENAI_CHAT] == 3
        assert got["unrecognised"] == 1


class TestAnAnthropicRecordIsNeverReadAsOpenAI:
    """The two formats spell the fields the same way and mean opposite things.

    Anthropic's `input_tokens` excludes the cached prefix; the OpenAI Responses
    API's includes it. `sniff` separated them on `cache_creation_input_tokens`
    alone, which a turn that *read* a cached prefix and wrote nothing does not
    carry -- and any proxy that omits zero-valued fields emits exactly that.

    Such a record sniffed as OpenAI, and the Responses adapter looks for the
    cached count under `input_tokens_details.cached_tokens`. It found none, so
    the whole cached prefix was dropped: a 500,000-token carry reported as
    zero, which on a long session is ~76% of the bill, with every number on
    screen still looking plausible.
    """

    @staticmethod
    def _anthropic(**usage):
        return {"model": "claude-opus-5",
                "usage": {"input_tokens": 10, "output_tokens": 50, **usage}}

    def test_a_read_only_turn_is_anthropic(self):
        rec = self._anthropic(cache_read_input_tokens=500_000)
        assert ingest.sniff(rec) == ingest.ANTHROPIC_API

    def test_and_its_cached_prefix_survives(self):
        u = ingest.usage_from(self._anthropic(cache_read_input_tokens=500_000))
        assert u.cache_read == 500_000
        assert u.uncached_in == 10

    def test_a_write_only_turn_is_still_anthropic(self):
        u = ingest.usage_from(self._anthropic(cache_creation_input_tokens=400_000))
        assert u.cache_write == 400_000

    def test_a_turn_with_both_fields_is_anthropic(self):
        rec = self._anthropic(cache_read_input_tokens=1, cache_creation_input_tokens=2)
        assert ingest.sniff(rec) == ingest.ANTHROPIC_API

    def test_a_genuine_responses_record_is_still_openai(self):
        rec = {"model": "gpt-5", "usage": {
            "input_tokens": 50_000, "output_tokens": 100,
            "input_tokens_details": {"cached_tokens": 48_000}}}
        assert ingest.sniff(rec) == ingest.OPENAI_RESPONSES

    def test_and_its_overlap_is_still_subtracted(self):
        """The conversion the module exists for must not regress."""
        u = ingest.usage_from({"model": "gpt-5", "usage": {
            "input_tokens": 50_000, "output_tokens": 100,
            "input_tokens_details": {"cached_tokens": 48_000}}})
        assert (u.uncached_in, u.cache_read) == (2_000, 48_000)

    def test_the_two_conventions_do_not_agree_by_accident(self):
        """Guards the premise: read one as the other and the context doubles."""
        anthropic = ingest.usage_from(self._anthropic(cache_read_input_tokens=500_000))
        assert anthropic.uncached_in + anthropic.cache_read == 500_010


class TestResponsesToolCallsAreRead:
    """A format whose tool calls are not extracted reads as a workload with none.

    Chat Completions puts calls under `choices[].message.tool_calls`; the
    Responses API puts them as `function_call` items in a flat `output` array.
    The Responses adapter extracted neither, so every such turn arrived with an
    empty `tools` tuple and `adder tools` / `trace --by tool` filed all of it
    under "(no tool call)" -- which reads as a finding about the workload rather
    than a gap in the reader.
    """

    @staticmethod
    def _rec(*names, **kw):
        return {"model": "gpt-5", "usage": {"input_tokens": 100, "output_tokens": 20},
                "output": [{"type": "function_call", "name": n} for n in names],
                **kw}

    def test_a_single_call_is_read(self):
        assert ingest.usage_from(self._rec("read_file")).tools == ("read_file",)

    def test_parallel_calls_are_all_read(self):
        assert ingest.usage_from(self._rec("read_file", "run_bash")).tools == (
            "read_file", "run_bash")

    def test_repeats_collapse_to_distinct_names(self):
        """`Turn.tools` is a set of names, as the Claude Code reader produces."""
        assert ingest.usage_from(self._rec("read_file", "read_file")).tools == (
            "read_file",)

    def test_non_call_output_items_are_ignored(self):
        rec = self._rec("read_file")
        rec["output"].append({"type": "message", "content": "hello"})
        assert ingest.usage_from(rec).tools == ("read_file",)

    def test_a_turn_with_no_output_field_has_no_tools(self):
        assert ingest.usage_from(
            {"model": "gpt-5",
             "usage": {"input_tokens": 1, "output_tokens": 1}}).tools == ()

    def test_chat_completions_still_works(self):
        rec = {"model": "gpt-5", "usage": {"prompt_tokens": 1, "completion_tokens": 1},
               "choices": [{"message": {"tool_calls": [
                   {"function": {"name": "read_file"}}]}}]}
        assert ingest.usage_from(rec).tools == ("read_file",)


class TestACoercionThatNeverRaises:
    """`_int` promises "Never raises on a log field" and raised on two of them.

    `int(inf)` is `OverflowError` and `int(nan)` is `ValueError`; only the
    second was caught. `json.loads` accepts the `Infinity` and `NaN` literals by
    default, so any proxy that serialises a float division emits a record that
    reaches this helper -- and one such field took down the read of the entire
    file, not just its own record.

    `catalog._num` already screened both out for the same reason. This is that
    screen applied to the package's other coercion helper.
    """

    @pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
    def test_a_non_finite_field_is_zero_not_an_exception(self, bad):
        assert ingest._int(bad) == 0

    @pytest.mark.parametrize("v,want", [
        (None, 0), ("", 0), ("abc", 0), (True, 0), (False, 0), ([], 0), ({}, 0),
        (-5, 0), (0, 0), (12, 12), ("12", 12), (1.9, 1),
    ])
    def test_the_ordinary_cases_are_unchanged(self, v, want):
        assert ingest._int(v) == want

    def test_a_decimal_string_now_parses_instead_of_becoming_zero(self):
        """`int("12.7")` raises; a log field written as a float should not vanish."""
        assert ingest._int("12.7") == 12
        assert ingest._int("1e5") == 100_000

    def test_a_record_carrying_infinity_still_yields_a_usage(self):
        u = ingest.usage_from({"model": "gpt-5", "usage": {
            "prompt_tokens": float("inf"), "completion_tokens": 10}})
        assert u is not None and u.uncached_in == 0 and u.out == 10

    def test_one_bad_line_does_not_take_down_the_file(self, tmp_path):
        p = tmp_path / "log.jsonl"
        p.write_text("\n".join([
            json.dumps({"model": "gpt-5",
                        "usage": {"prompt_tokens": 100, "completion_tokens": 10}}),
            '{"model":"gpt-5","usage":{"prompt_tokens":Infinity,"completion_tokens":10}}',
            json.dumps({"model": "gpt-5",
                        "usage": {"prompt_tokens": 200, "completion_tokens": 20}}),
        ]))
        turns = list(ingest.iter_turns(p))
        assert [t.uncached_in for t in turns] == [100, 0, 200]


class TestForeignLogsThatDoNotMatchTheirOwnShape:
    """A record that sniffs as one format but is malformed inside it.

    Each adapter's contract is to normalise whatever it is handed. Reaching for
    `.get` on a field that turned out to be a string or a number raised
    `AttributeError` out of that contract, and `iter_turns` has no handler --
    so one bad record ended the read of the whole log.
    """

    def test_a_string_usage_on_an_anthropic_record(self):
        from adder.core.ingest import usage_from

        assert usage_from({"type": "message", "model": "claude-opus-5",
                           "usage": "lots"}) is None

    def test_a_scalar_token_details_block(self):
        from adder.core.ingest import usage_from

        got = usage_from({"model": "gpt-5",
                          "usage": {"prompt_tokens": 100,
                                    "prompt_tokens_details": 7}})
        assert got.uncached_in == 100 and got.cache_read == 0

    def test_a_scalar_usage_metadata_on_a_gemini_record(self):
        from adder.core.ingest import usage_from

        assert usage_from({"usageMetadata": 5}) is None

    def test_a_non_string_model_never_reaches_the_registry(self):
        from adder.core.ingest import usage_from

        assert usage_from({"usage": {"prompt_tokens": 10},
                           "model": {"name": "x"}}) is None


class TestForeignDedupKeepsPosition:
    """Two identical calls are two calls, and the later record replaces its own.

    `order.index(prev)` compared `Turn`s by value, so a log holding two
    byte-identical calls had the *first* one replaced when the *second* one's
    stream completed -- silently moving output tokens between turns, and doing
    it in quadratic time.
    """

    def test_identical_calls_are_both_kept(self, tmp_path):
        import json

        from adder.core.ingest import iter_turns

        def rec(mid, out):
            return {"model": "gpt-5", "id": mid,
                    "usage": {"prompt_tokens": 100, "completion_tokens": out}}

        path = tmp_path / "log.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in [
            rec("a", 10), rec("b", 10), rec("b", 50), rec("a", 20)]))
        turns = list(iter_turns(path))
        assert [(t.msg_id, t.out) for t in turns] == [("a", 20), ("b", 50)]


class TestClassifyFiles:
    """Per-file provenance, which is the grain a person asks the question at.

    A mis-sniffed log does not error: it reads the cached prefix under the
    wrong convention and reports a plausible bill roughly double the real one.
    So which adapter fired has to be visible, and `adder doctor` shows it.
    """

    def _write(self, p, rows):
        p.write_text("\n".join(json.dumps(r) for r in rows))

    def test_each_file_is_named_by_its_own_shape(self, tmp_path):
        self._write(tmp_path / "a.jsonl", [
            {"id": "1", "model": "gpt-5",
             "usage": {"prompt_tokens": 10, "completion_tokens": 2}}])
        self._write(tmp_path / "b.jsonl", [
            {"modelVersion": "gemini-3-pro",
             "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 2}}])
        tally, silent = ingest.classify_files(sorted(tmp_path.glob("*.jsonl")))
        assert tally == {ingest.OPENAI_CHAT: 1, ingest.GEMINI: 1}
        assert silent == []

    def test_a_file_with_no_usage_is_reported_rather_than_dropped(self, tmp_path):
        self._write(tmp_path / "chat.jsonl", [
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "yo"}])
        tally, silent = ingest.classify_files(sorted(tmp_path.glob("*.jsonl")))
        assert tally == {}
        assert [p.name for p in silent] == ["chat.jsonl"]

    def test_the_sample_cap_is_honoured(self, tmp_path):
        # Only the head of a file is inspected, which is what keeps this
        # affordable on every `adder doctor`.
        rows = [{"role": "user", "content": "x"}] * 5
        rows += [{"id": "z", "model": "gpt-5",
                  "usage": {"prompt_tokens": 10, "completion_tokens": 2}}]
        self._write(tmp_path / "late.jsonl", rows)
        tally, silent = ingest.classify_files(sorted(tmp_path.glob("*.jsonl")), sample=3)
        assert tally == {} and len(silent) == 1
        tally, silent = ingest.classify_files(sorted(tmp_path.glob("*.jsonl")), sample=50)
        assert tally == {ingest.OPENAI_CHAT: 1} and silent == []


class TestRecordsToolCalls:
    """A zero that means "unrecorded" must not print as a zero that means "none".

    `adder context` on a Gemini log showed "tool results 0.0%". On the corpus
    adder was built from, `Bash` results alone were 80% of context growth, so
    that zero is the most expensive wrong impression the tool can leave.
    """

    def test_claude_code_records_tool_calls(self, tmp_path):
        (tmp_path / "s.jsonl").write_text(json.dumps({
            "type": "assistant",
            "message": {"id": "m1", "model": "claude-opus-5",
                        "usage": {"input_tokens": 10, "output_tokens": 2}}}))
        assert ingest.records_tool_calls(tmp_path)

    def test_a_usage_only_format_does_not(self, tmp_path):
        (tmp_path / "s.jsonl").write_text(json.dumps({
            "id": "1", "model": "gpt-5",
            "usage": {"prompt_tokens": 10, "completion_tokens": 2}}))
        assert not ingest.records_tool_calls(tmp_path)

    def test_a_mixed_corpus_keeps_the_detail(self, tmp_path):
        # One Claude Code file is enough for the breakdown to mean something,
        # so the caveat would be the misleading half here.
        (tmp_path / "a.jsonl").write_text(json.dumps({
            "id": "1", "model": "gpt-5",
            "usage": {"prompt_tokens": 10, "completion_tokens": 2}}))
        (tmp_path / "b.jsonl").write_text(json.dumps({
            "type": "assistant",
            "message": {"id": "m1", "model": "claude-opus-5",
                        "usage": {"input_tokens": 10, "output_tokens": 2}}}))
        assert ingest.records_tool_calls(tmp_path)

    def test_an_unreadable_root_does_not_add_a_caveat(self, tmp_path):
        # Unknown is not the same as "no tool detail"; a caveat asserted over
        # nothing is its own wrong claim.
        assert ingest.records_tool_calls(tmp_path / "does-not-exist")
        assert ingest.records_tool_calls(tmp_path)


class TestHeadRecordsIsBounded:
    """Sniffing a file must not read it.

    `classify_files` asked for fifty records per file and got them through
    `iter_records`, which calls `read_text()` first. On 278 transcripts that
    read every byte of every one and put 22 seconds onto `adder doctor` --
    a cap enforced at the consumer is not a cap.
    """

    def test_it_yields_no_more_than_the_cap(self, tmp_path):
        p = tmp_path / "big.jsonl"
        p.write_text("\n".join(json.dumps({"i": i}) for i in range(1000)))
        assert len(list(ingest._head_records(p, 5))) == 5

    def test_a_jsonl_file_never_goes_through_the_whole_file_reader(
            self, tmp_path, monkeypatch):
        p = tmp_path / "s.jsonl"
        p.write_text("\n".join(json.dumps({
            "id": str(i), "model": "gpt-5",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
            for i in range(50)))

        def _boom(*_a, **_k):
            raise AssertionError("read the whole file to sniff it")

        monkeypatch.setattr(ingest, "iter_records", _boom)
        assert len(list(ingest._head_records(p, 3))) == 3

    def test_a_pretty_printed_object_still_reads(self, tmp_path):
        # The fallback that keeps a valid single-object log from being called
        # unreadable: no line of it parses on its own.
        p = tmp_path / "one.json"
        p.write_text(json.dumps(
            {"id": "1", "model": "gpt-5",
             "usage": {"prompt_tokens": 10, "completion_tokens": 2}}, indent=2))
        assert len(list(ingest._head_records(p, 5))) == 1

    def test_a_json_array_still_reads(self, tmp_path):
        p = tmp_path / "arr.json"
        p.write_text(json.dumps([
            {"id": str(i), "model": "gpt-5",
             "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
            for i in range(20)]))
        assert len(list(ingest._head_records(p, 4))) == 4

    def test_classify_stops_once_a_shape_is_settled(self, tmp_path):
        # The early exit must not change the answer, only the work.
        rows = [{"id": str(i), "model": "gpt-5",
                 "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
                for i in range(200)]
        (tmp_path / "a.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows))
        tally, silent = ingest.classify_files([tmp_path / "a.jsonl"])
        assert tally == {ingest.OPENAI_CHAT: 1} and silent == []
