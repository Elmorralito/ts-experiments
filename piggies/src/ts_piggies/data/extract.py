from abc import ABC, abstractmethod
from datetime import datetime
import itertools
import re
from pydantic import BaseModel, ConfigDict, PrivateAttr
from typing import Any, Dict, Iterable, List, Literal, Sequence, Self, final
import pandas as pd
from .model import PriceData
from .downloader import Downloader
from .pdf import AbstractPDFParser, PDF2TextParser

class AbstractExtractor(BaseModel, ABC):
    """Abstract base class for data extractors."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    overwrite: Literal["ALL", "FILE", "DIRECTORY", "NONE"] | bool = True
    downloader: Downloader

    _prepare_result: Iterable[Any] = PrivateAttr(default_factory=lambda: iter([]))
    _extract_result: pd.DataFrame = PrivateAttr(default_factory=pd.DataFrame)

    @abstractmethod
    def prepare(self, url: str, *urls: Sequence[str]) -> Self:
        """Download data from a source."""

    @abstractmethod
    def extract_data(self) -> List[PriceData]:
        """Extract data from a source."""

    @abstractmethod
    def extract(self) -> Self:
        """Extract data from a source."""

    @property
    def result(self) -> pd.DataFrame:
        """Get the result of the extraction."""
        return self._extract_result

    def reset(self) -> Self:
        """Clean the data."""
        self._prepare_result = []
        self._extract_result = pd.DataFrame()
        return self

    def run(self, url: str, *urls: Sequence[str]) -> Self:
        """Run the extraction."""
        return self.reset().prepare(url, *urls).extract()


class PDFExtractor(AbstractExtractor):
    """Extractor for PDF files."""

    parser: AbstractPDFParser

    def prepare(self, url: str, *urls: Sequence[str]) -> Self:
        """Download PDF files from a source."""
        paths = self.downloader.download(url, *urls, overwrite=self.overwrite).result
        self._prepare_result = list(itertools.chain(*[self.parser.parse(path) for path in paths]))
        return self
    
    def extract(self) -> Self:
        """Extract data from a source."""
        raise NotImplementedError("Extract method not implemented")


class PorkColombiaPDFExtractor(PDFExtractor):
    """Extractor for Pork Colombia PDF files."""

    MONTHS_NAMES_MAPPER: Dict[str, str] = {
        "enero": "january",
        "febrero": "february",
        "marzo": "march",
        "abril": "april",
        "mayo": "may",
        "junio": "june",
        "julio": "july",
        "agosto": "august",
        "septiembre": "september",
        "octubre": "october",
        "noviembre": "november",
        "diciembre": "december",
    }
    DEFAULT_MARKET: str = "porkcolombia"

    def prepare(self, url: str, *urls: Sequence[str]) -> Self:
        if not isinstance(self.parser, PDF2TextParser):
            raise ValueError("Parser must be a PDF2TextParser")

        return super().prepare(url, *urls)

    def extract_date(self, page: str) -> datetime:
        date_raw = re.search(r"\n(\w+)\s+(\w+)\s+de\s+\w+", page).group().strip()
        month_raw = date_raw.split()[0]
        month = self.MONTHS_NAMES_MAPPER[month_raw.lower()]
        date_ = date_raw.replace(month_raw, month).replace("de ", "").strip()
        return datetime.strptime(date_, "%B %d %Y")

    def extract_data(self, page: str) -> List[PriceData]:
        content = re.search(r"Antioquia (\d+[\.|\,]\d+) (\d+[\.|\,]\d+) (\d+[\.|\,]\d+)", page)
        if content is None:
            raise ValueError("No price data found")

        ts = self.extract_date(page)
        return [
            PriceData(
                ts=ts,
                minimum=content.group(1),
                average=content.group(2),
                maximum=content.group(3),
                market=self.DEFAULT_MARKET,
            )
        ]

    def locate_page(self, page: str) -> bool:
        checkmark = "PRECIOS PROMEDIOS (Pagados al porcicultor *)"
        return page.find(checkmark) != -1

    def extract(self) -> Self:
        """Extract data from a source."""
        desired_pages = []
        for page in self._prepare_result:
            if not self.locate_page(page):
                continue

            data = self.extract_data(page)
            desired_pages.extend([d.model_dump(mode="python") for d in data])

        if not desired_pages:
            raise ValueError("No target pages found")

        self._extract_result = pd.DataFrame(desired_pages).set_index("ts").sort_index(ascending=False)
        return self
