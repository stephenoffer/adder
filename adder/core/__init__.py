"""Reading a Claude Code session off disk, and the settings that govern it.

`trace` is the only module that parses a transcript, so the deduplication rule
that every number depends on lives in exactly one place. Everything above this
package consumes `Session` and `Turn`; nothing in it may import upward.
"""
