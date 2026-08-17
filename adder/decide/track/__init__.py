"""The record of past decisions, and whether they paid.

Recommendations are cheap to make and expensive to trust. These modules keep the
outcome log that calibrates `p_fail`, the ledger that asks whether the advice has
been worth more than the asking, and the delegations recovered from transcripts
that say what was actually routed rather than what was suggested.
"""
