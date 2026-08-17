"""Re-running recorded work under a counterfactual.

Every module here answers a question of the form "what would this have cost if
X". They share one hard rule: the baseline is the turns that actually happened,
priced by the same cost model as the report -- a harness that invents its own
baseline can prove anything.
"""
