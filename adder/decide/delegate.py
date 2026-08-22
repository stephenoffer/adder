"""The step before a delegation: what the delegated work should run on.

`route/policy.py` has answered this since the beginning, and answered it well —
it classifies the task, prices every rung of the ladder including the chance of
having to redo the work, and refuses to recommend anything whose saving does not
clear the turn spent asking. What it never had was a moment. It runs when a human
types `adder policy "<task>"`, and nothing typed it at the point where the
decision is actually taken, which is the `Task` call itself. A router nobody
invokes routes nothing, and the tier agents `adder auto on` installs are a *fixed*
assignment: whatever the caller names, or the session model if it names nothing.

So this is the same decision, moved to the moment. The PreToolUse guard already
sees every `Task` and already prices what the subagent hands *back*; this adds
what it should run *on*. One hook, one message, one ledger entry, because two
sentences injected about one call cost twice as much to carry as one.

Four rules, each of them a test, and each of them a way this could go wrong:

* **It never refuses a delegation.** The guard may refuse a read; refusing a
  `Task` would be refusing the single largest lever this tool argues for. Being
  wrong here costs a sentence, so a sentence is all it gets to be.
* **It says nothing when the caller has already decided.** A call that names
  `route-t1` or `Explore` carries a routing decision already. Re-litigating it
  is noise, and noise is billed on every remaining turn.
* **It prices its own sentence, discounted by whether anyone listens.** This is
  advice, not enforcement, so the saving is worth `advice_taken` of itself and
  has to beat the cost of carrying the words. That is the same test
  `guard._ledger_gate` applies, deliberately.
* **The comparison is against what this delegation would otherwise have run
  on**, which under Claude Code is the session model. Not against the most
  expensive rung, and not against inline: the placement decision has already
  been taken by the caller, and re-pricing it would quote a saving from a lever
  somebody already pulled.

Where the arena data enters, and where it does not
-------------------------------------------------
`route/select.py` ranks ~500 catalogued models by price against LMArena Elo, and
`policy.decide` already reports the cross-vendor ones as substitutes. None of
them appear in the message this module builds, because under Claude Code a
`Task` cannot be dispatched to Qwen: the harness pins subagents to the vendor
(`core/harness.py`), so quoting one here would be advice nobody can take at the
exact moment they cannot take it. The arena signal reaches this decision the one
way it can be acted on — through `ladder()`, which is what `adder models ladder`
diffs against the live catalog, and through `p_fail` where the outcome log is
still thin. Naming a model the harness cannot run is how a router loses trust.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from adder.pricing.cost import admitted_token_cost
from adder.util.text import est_tokens

# A `subagent_type` in this set has already been routed, by a caller who said so.
# `Explore` is included because `auto on` installs it pointed at the cheapest
# rung: it is the one agent whose whole purpose is that it already runs cheap.
ROUTED: frozenset[str] = frozenset({
    'route-t0', 'route-t1', 'route-t2', 'route-t3', 'explore',
})
# How much of a task description to hand the classifier. It reads vocabulary,
# not structure, so the first paragraph carries the signal; the rest is prompt
# body that would only slow a hook down.
TASK_CHARS = 600
# A model id with the context suffix and any date stamp removed, for deciding
# whether two names are the same model. `claude-opus-5[1m]` and
# `claude-opus-5-20260214` are the same rung, and a tier recommendation that
# fires because the two strings differ is a recommendation to switch a model for
# itself -- at the price of a cache the switch would throw away.
_SUFFIX = re.compile(r'(\[[^\]]*\]|-\d{6,8})+$')


def stem(model: str) -> str:
    """The model, without the context suffix or date stamp. Never empty-guessing."""
    return _SUFFIX.sub('', (model or '').strip().lower())


@dataclass(frozen=True)
class Advice:
    """What to say about where a delegated step should run, or why to stay quiet."""

    fire: bool
    reason: str
    message: str = ''
    # The same claim in a clause rather than a sentence, for the case where this
    # is appended to a message the guard was already sending. Words are billed:
    # a joined message that repeats "in expectation, including a chance of
    # having to redo it" pays for the phrase twice and risks being clipped
    # before the part that says what to do.
    clause: str = ''
    agent: str = ''
    model: str = ''
    was: str = ''
    saving: float = 0.0
    overhead: float = 0.0
    advice_taken: float = 0.5
    target: str = ''

    @property
    def net(self) -> float:
        """Expected value of saying it. Discounted: this is advice, not a refusal."""
        return self.saving * self.advice_taken - self.overhead


def task_text(tool_input: dict) -> str:
    """The task, as the classifier should see it.

    `description` is the one-line summary the harness asks for and `prompt` is
    the body; a caller may send either. Joined rather than chosen between,
    because the vocabulary the classifier keys on ("refactor", "across") turns up
    in whichever one the caller bothered to write.
    """
    tool_input = tool_input or {}
    parts = [str(tool_input.get(k) or '').strip()
             for k in ('description', 'prompt', 'task')]
    return ' '.join(p for p in parts if p)[:TASK_CHARS].strip()


def advise(tool_input: dict, *, session_model: str, remaining_turns: int,
           context_tokens: int, carry=None, project: str | None = None,
           advice_taken: float = 0.5, min_cost: float = 0.0, plan=None) -> Advice:
    """Should anything be said about the tier this delegation will run on?

    `plan` is injectable so the gates can be tested without an outcome log or a
    catalog on disk; left as `None` it asks `policy.decide`, which is the only
    expensive thing here and is reached only after the cheap refusals above it.
    """
    sub = str((tool_input or {}).get('subagent_type') or '').strip()
    if sub.lower() in ROUTED:
        return Advice(False, f'{sub} already names a routed subagent')
    task = task_text(tool_input)
    if not task:
        return Advice(False, 'no task text to classify; refusing to guess a tier')

    if plan is None:
        from adder.decide.route.policy import decide
        plan = decide(task, context_tokens=context_tokens,
                      remaining_turns=remaining_turns, session_model=session_model,
                      project=project, carry=carry)
    chosen = next((r for r in plan.ladder if r.tier is plan.tier), None)
    if chosen is None or not plan.agent:
        return Advice(False, 'no priced ladder to compare against')

    # The baseline is the rung this call would have run on anyway. If the session
    # model is not on the ladder there is no honest comparison to draw -- the
    # difference between two tiers is a number, the difference between a tier and
    # an unpriced unknown is not -- so the answer is silence, not the top rung.
    base = next((r for r in plan.ladder if stem(r.model) == stem(session_model)), None)
    if base is None:
        return Advice(False, f'{session_model} is not on the ladder; nothing to compare')
    if base.tier is chosen.tier:
        return Advice(False, 'this would already run on the cheapest tier in expectation')

    saving = base.expected - chosen.expected
    if saving <= 0:
        return Advice(False, 'the tier it would already use is no more expensive')
    if saving < min_cost:
        return Advice(False, f'${saving:,.2f} cheaper, below the ${min_cost:.2f} floor')

    # One clause, and the p_fail is in it on purpose: a cheaper tier is only
    # cheaper net of the chance of redoing the work, and a reader who cannot see
    # that number cannot check the claim.
    why = next((r for r in plan.reasons if r), '')
    msg = (f'[adder] Run this on {plan.agent} ({chosen.model}) rather than '
           f'{session_model}: ~${saving:,.2f} cheaper in expectation, including a '
           f'{chosen.p_fail:.0%} chance of having to redo it')
    msg += f' ({why}).' if why else '.'
    clause = (f'[adder] Run it on {plan.agent} ({chosen.model}): ~${saving:,.2f} '
              f'cheaper, redo risk included.')
    over = admitted_token_cost(est_tokens(msg), session_model, remaining_turns,
                               carry=carry, context_tokens=context_tokens)
    out = Advice(True, 'a cheaper tier clears the cost of saying so', message=msg,
                 clause=clause,
                 agent=plan.agent, model=chosen.model, was=session_model,
                 saving=saving, overhead=over, advice_taken=advice_taken,
                 target=f'Task:tier:{chosen.tier.name}')
    if out.net <= 0:
        return Advice(False, f'saying so costs ${over:,.4f} to carry and is worth '
                             f'${saving * advice_taken:,.4f}')
    return out
