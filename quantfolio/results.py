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

    def table(self, heading: str) -> pd.DataFrame:
        for s in self.sections:
            if s.heading == heading and isinstance(s.body, pd.DataFrame):
                return s.body
        raise KeyError(heading)
