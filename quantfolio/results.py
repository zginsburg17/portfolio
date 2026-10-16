"""Structured study output, so studies can be tested, saved or plotted without parsing stdout."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Section:
    heading: str
    body: pd.DataFrame | str


@dataclass
class StudyResult:
    title: str
    sections: list[Section] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)  # headline numbers, machine-readable

    def add(self, heading: str, body: pd.DataFrame | str) -> None:
        self.sections.append(Section(heading, body))

    def table(self, heading_prefix: str) -> pd.DataFrame:
        """The first table section whose heading starts with `heading_prefix`."""
        for s in self.sections:
            if s.heading.startswith(heading_prefix) and isinstance(s.body, pd.DataFrame):
                return s.body
        raise KeyError(heading_prefix)

    def text(self, heading_prefix: str) -> str:
        for s in self.sections:
            if s.heading.startswith(heading_prefix) and isinstance(s.body, str):
                return s.body
        raise KeyError(heading_prefix)
