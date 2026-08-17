"""Checking a stated claim against new data.

The README makes numeric claims and the gates act on them. These modules are how
those claims stay honest: each one re-derives a published figure from whatever
transcripts are on the machine now, and says plainly when the figure no longer
holds. If a claim changes, the module that tests it changes in the same commit.
"""
