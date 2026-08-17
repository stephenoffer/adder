"""What a token costs, across providers.

Split three ways on purpose: `prices` is the hand-maintained first-party Claude
table, `catalog` is the scraped cross-provider snapshot, and `providers` holds
the billing mechanics that decide how caching is paid for. `registry` joins
them so callers ask one question instead of three.
"""
