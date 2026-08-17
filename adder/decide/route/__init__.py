"""Choosing where a task runs, and on what.

The order these run in is the design: classify the task, price the candidates
against the session's actual state, then check that the recommendation clears
its own overhead. A module here that skips the last step is a module that will
confidently advise a switch costing more than it saves.
"""
