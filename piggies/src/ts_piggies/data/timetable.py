

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable, List
import pandas as pd
from pydantic import BaseModel
from .model import URL


class AbstractTimetable(BaseModel, ABC):
    """Abstract base class for timetables."""

    start_date: datetime
    end_date: datetime
    url_template: str
    url_fallback_template: str = ""

    @property
    @abstractmethod
    def timetable(self) -> pd.DataFrame:
        """Get the timetable."""

    @property
    @abstractmethod
    def urls(self) -> Iterable[URL]:
        """Build the URLs for the timetable."""

    @abstractmethod
    def build_url(self, timerow: pd.Series) -> URL:
        """Build the URL for the timetable."""


class PorkColombiaTimetable(AbstractTimetable):
    """Timetable for Pork Colombia."""

    start_date: datetime
    end_date: datetime
    url_template: str = "https://porkcolombia.co/wp-content/uploads/{year}/{month:02d}/Semana{week:02d}de{year}.pdf"
    url_fallback_templates: List[str] = [
        "https://porkcolombia.co/wp-content/uploads/{year}/{month:02d}/Semana{week:02d}de{year}-1.pdf",
        "https://porkcolombia.co/wp-content/uploads/{year}/{month:02d}/Semana{week:02d}de{year}-2.pdf",
        "https://porkcolombia.co/wp-content/uploads/{year}/{month:02d}/Informe_Pork_Colombia_Ronda{week:02d}.pdf",
        "https://porkcolombia.co/wp-content/uploads/{year}/{month:02d}/Informe_Pork_Colombia_Ronda{week:02d}-1.pdf",
        "https://porkcolombia.co/wp-content/uploads/{year}/{month:02d}/Informe_Pork_Colombia_Ronda{week:02d}-2.pdf",
    ]

    @property
    def timetable(self) -> pd.DataFrame:
        """Get the timetable."""
        time_range = pd.date_range(start=self.start_date, end=self.end_date, freq="W")
        timetable = pd.DataFrame(time_range, columns=["ts"]).sort_values(by="ts", ascending=False)
        timetable["year"] = timetable["ts"].dt.year
        timetable["month"] = timetable["ts"].dt.month
        timetable["week"] = timetable["ts"].dt.isocalendar().week
        return timetable

    @property
    def urls(self) -> Iterable[URL]:
        """Build the URLs for the timetable."""
        return self.timetable.apply(self.build_url, axis=1).tolist()

    def build_url(self, timerow: pd.Series) -> URL:
        """Build the URL for the timetable."""
        timedict_ = timerow.to_dict()
        timedict = {
            "year": timedict_["year"],
            "month": timedict_["month"],
            "week": timedict_["week"],
        }
        return URL(
            url=self.url_template.format(**timedict),
            fallback_urls=[template.format(**timedict) for template in self.url_fallback_templates]
        )
