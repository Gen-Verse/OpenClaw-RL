"""
openclaw-training-free — Training-Free LLM improvement via experience loops.

A lightweight system that improves LLM responses over time *without* updating
model parameters (training-free). It intercepts conversations, extracts
improvement hints via a Judge LLM, stores them in a keyword-indexed experience
library, and injects relevant past experiences into future prompts — forming a
self-improving experience loop.

Compared to openclaw-rl (~$10,000/run) and openclaw-opd (~$18/run), this
approach costs ~$0 in training compute.

Main components:

- :class:`ExperienceStore`     — persist and retrieve hints (JSON file).
- :class:`ExperienceExtractor` — call a Judge LLM to extract hints.
- :class:`PromptInjector`      — inject retrieved hints into prompts.

See ``training_free_server.py`` for the FastAPI proxy that ties them together.
"""

from __future__ import annotations

from experience_store import ExperienceStore
from experience_extractor import ExperienceExtractor
from prompt_injector import PromptInjector

__all__ = [
    "ExperienceStore",
    "ExperienceExtractor",
    "PromptInjector",
]
