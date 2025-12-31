import re
from typing import Iterable, List, Self
from pydantic import BaseModel, Field, field_validator, model_validator

from datetime import datetime


class PriceData(BaseModel):

    ts: datetime
    minimum: float | str
    maximum: float | str
    average: float | str | None = None
    market: str | None = None

    @field_validator("minimum", "maximum", "average")
    @classmethod
    def _validate_float(cls, v: float | str | None) -> float | None:
        if v is None or isinstance(v, float):
            return v

        v = re.sub(r"[^\.\,\d]", "", v)
        try:
            return float(v.replace(".", ""))
        except ValueError:
            return None

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        minimum = self.minimum
        maximum = self.maximum
        self.minimum = min(minimum, maximum)
        self.maximum = max(minimum, maximum)
        if self.average is None:
            self.average = (self.minimum + self.maximum) / 2

        return self


class URL(BaseModel):
    url: str
    fallback_urls: List[str] = Field(default_factory=list)

    @property
    def urls(self) -> Iterable[str]:
        """Build the URLs for the timetable."""
        return [self.url, *self.fallback_urls]
