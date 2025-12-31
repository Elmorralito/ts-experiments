"""PDF parsing module with abstract base class and implementations."""
from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Literal

# PDF processing libraries
import pypdf
import pdfplumber
import pytesseract
import pdf2image

logger = logging.getLogger(__name__)


class AbstractPDFParser(ABC):
    """Abstract base class for PDF parsing implementations."""

    @abstractmethod
    def parse(self, pdf_path: str | Path) -> list[str]:
        """
        Parse a PDF file and extract text content.

        Args:
            pdf_path: Path to the PDF file to parse.

        Returns:
            List of extracted text content, one string per page.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the file is not a valid PDF.
            Exception: For other parsing errors.
        """
        pass

    @abstractmethod
    def parse_page(self, pdf_path: str | Path, page_number: int) -> str:
        """
        Parse a specific page from a PDF file.

        Args:
            pdf_path: Path to the PDF file to parse.
            page_number: Zero-based page number to extract.

        Returns:
            Extracted text content from the specified page.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the file is not a valid PDF or page number is invalid.
            IndexError: If page number is out of range.
            Exception: For other parsing errors.
        """
        pass

    @abstractmethod
    def get_page_count(self, pdf_path: str | Path) -> int:
        """
        Get the total number of pages in a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Total number of pages in the PDF.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the file is not a valid PDF.
            Exception: For other parsing errors.
        """
        pass


class PDF2TextParser(AbstractPDFParser):
    """
    Implementation of PDF2TextParser supporting multiple extraction methods.

    Supports:
    - Direct text extraction (using pdfplumber)
    - OCR-based extraction (using pytesseract)
    """

    __slots__ = ("method", "ocr_lang", "fallback_to_ocr")

    def __init__(
        self,
        method: Literal["text", "ocr", "auto"] = "auto",
        ocr_lang: str = "eng",
        fallback_to_ocr: bool = True,
    ) -> None:
        """
        Initialize the PDF parser.

        Args:
            method: Extraction method to use.
                  - "text": Direct text extraction (faster, works for text-based PDFs)
                  - "ocr": OCR-based extraction (slower, works for scanned PDFs)
                  - "auto": Try text extraction first, fallback to OCR if needed
            ocr_lang: Language code for OCR (e.g., "eng", "spa"). Defaults to "eng".
            fallback_to_ocr: Whether to fallback to OCR if text extraction fails.
                           Only used when method="auto". Defaults to True.
        """
        if method not in ("text", "ocr", "auto"):
            raise ValueError(f"Invalid method: {method}. Must be 'text', 'ocr', or 'auto'")

        self.method = method
        self.ocr_lang = ocr_lang
        self.fallback_to_ocr = fallback_to_ocr
        logger.info(
            f"Initialized {self.__class__.__name__} with method={method}, ocr_lang={ocr_lang}, "
            f"fallback_to_ocr={fallback_to_ocr}"
        )

    def parse(self, pdf_path: str | Path) -> list[str]:
        """
        Parse a PDF file and extract text content.

        Args:
            pdf_path: Path to the PDF file to parse.

        Returns:
            List of extracted text content, one string per page.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the file is not a valid PDF.
            Exception: For other parsing errors.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if not pdf_path.is_file():
            raise ValueError(f"Path is not a file: {pdf_path}")

        logger.debug(f"Parsing PDF: {pdf_path} with method={self.method}")

        if self.method == "text":
            return self._extract_text(pdf_path)
        elif self.method == "ocr":
            return self._extract_with_ocr(pdf_path)
        else:  # method == "auto"
            try:
                return self._extract_text(pdf_path)
            except Exception as e:
                logger.warning(f"Text extraction failed for {pdf_path}: {e}")
                if self.fallback_to_ocr:
                    logger.info(f"Falling back to OCR for {pdf_path}")
                    return self._extract_with_ocr(pdf_path)
                raise

    def parse_page(self, pdf_path: str | Path, page_number: int) -> str:
        """
        Parse a specific page from a PDF file.

        Args:
            pdf_path: Path to the PDF file to parse.
            page_number: Zero-based page number to extract.

        Returns:
            Extracted text content from the specified page.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the file is not a valid PDF or page number is invalid.
            IndexError: If page number is out of range.
            Exception: For other parsing errors.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if page_number < 0:
            raise ValueError(f"Page number must be non-negative, got {page_number}")

        total_pages = self.get_page_count(pdf_path)
        if page_number >= total_pages:
            raise IndexError(
                f"Page number {page_number} is out of range. PDF has {total_pages} pages."
            )

        logger.debug(f"Parsing page {page_number} from PDF: {pdf_path}")

        if self.method == "text":
            return self._extract_text_page(pdf_path, page_number)
        elif self.method == "ocr":
            return self._extract_with_ocr_page(pdf_path, page_number)
        else:  # method == "auto"
            try:
                return self._extract_text_page(pdf_path, page_number)
            except Exception as e:
                logger.warning(f"Text extraction failed for page {page_number} of {pdf_path}: {e}")
                if self.fallback_to_ocr:
                    logger.info(f"Falling back to OCR for page {page_number} of {pdf_path}")
                    return self._extract_with_ocr_page(pdf_path, page_number)
                raise

    def get_page_count(self, pdf_path: str | Path) -> int:
        """
        Get the total number of pages in a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Total number of pages in the PDF.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the file is not a valid PDF.
            Exception: For other parsing errors.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            with open(pdf_path, "rb") as file:
                reader = pypdf.PdfReader(file)
                page_count = len(reader.pages)
                logger.debug(f"PDF {pdf_path} has {page_count} pages")
                return page_count
        except ImportError:
            logger.error("pypdf is not installed. Please install it to use PDF parsing.")
            raise ImportError("pypdf is required for PDF parsing. Install it with: pip install pypdf")
        except Exception as e:
            raise ValueError(f"Failed to read PDF file {pdf_path}: {str(e)}") from e

    def _extract_text(self, pdf_path: Path) -> list[str]:
        """Extract text directly from PDF using pdfplumber (preferred) or pypdf. Returns list of page texts."""
        try:
            logger.debug(f"Extracting text from {pdf_path} using pdfplumber")
            with pdfplumber.open(pdf_path) as pdf:
                text_parts = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    text_parts.append(page_text if page_text else "")
                return text_parts
        except ImportError:
            logger.debug("pdfplumber not available, falling back to pypdf")
            try:
                logger.debug(f"Extracting text from {pdf_path} using pypdf")
                with open(pdf_path, "rb") as file:
                    reader = pypdf.PdfReader(file)
                    text_parts = []
                    for page in reader.pages:
                        page_text = page.extract_text()
                        text_parts.append(page_text if page_text else "")
                    return text_parts
            except ImportError:
                raise ImportError(
                    "Neither pdfplumber nor pypdf is installed. "
                    "Install at least one: pip install pdfplumber pypdf"
                )
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF {pdf_path}: {str(e)}") from e

    def _extract_text_page(self, pdf_path: Path, page_number: int) -> str:
        """Extract text from a specific page using pdfplumber (preferred) or pypdf."""
        try:
            logger.debug(f"Extracting text from page {page_number} of {pdf_path} using pdfplumber")
            with pdfplumber.open(pdf_path) as pdf:
                if page_number >= len(pdf.pages):
                    raise IndexError(f"Page {page_number} is out of range")
                page = pdf.pages[page_number]
                text = page.extract_text()
                return text if text else ""
        except ImportError:
            logger.debug("pdfplumber not available, falling back to pypdf")
            try:
                logger.debug(f"Extracting text from page {page_number} of {pdf_path} using pypdf")
                with open(pdf_path, "rb") as file:
                    reader = pypdf.PdfReader(file)
                    if page_number >= len(reader.pages):
                        raise IndexError(f"Page {page_number} is out of range")
                    page = reader.pages[page_number]
                    text = page.extract_text()
                    return text if text else ""
            except ImportError:
                raise ImportError(
                    "Neither pdfplumber nor pypdf is installed. "
                    "Install at least one: pip install pdfplumber pypdf"
                )
        except Exception as e:
            raise ValueError(
                f"Failed to extract text from page {page_number} of PDF {pdf_path}: {str(e)}"
            ) from e

    def _extract_with_ocr(self, pdf_path: Path) -> list[str]:
        """Extract text from PDF using OCR. Returns list of page texts."""
        try:
            logger.debug(f"Extracting text from {pdf_path} using OCR")
            images = pdf2image.convert_from_path(pdf_path)
            text_parts = []
            for i, image in enumerate(images):
                logger.debug(f"Processing page {i + 1}/{len(images)} with OCR")
                page_text = pytesseract.image_to_string(image, lang=self.ocr_lang)
                text_parts.append(page_text if page_text else "")
            return text_parts
        except ImportError as e:
            missing_package = str(e).split()[-1] if "No module named" in str(e) else "unknown"
            raise ImportError(
                f"OCR dependencies not installed. Install with: "
                f"pip install pytesseract pillow pdf2image. "
                f"Also ensure Tesseract OCR is installed on your system. "
                f"Missing: {missing_package}"
            ) from e
        except Exception as e:
            error_msg = str(e)
            error_lower = error_msg.lower()
            
            if "poppler" in error_lower or "unable to get page count" in error_lower:
                raise RuntimeError(
                    f"Poppler is required for OCR but not found. "
                    f"Install it with:\n"
                    f"  macOS: brew install poppler\n"
                    f"  Ubuntu/Debian: sudo apt-get install poppler-utils\n"
                    f"  Fedora/RHEL: sudo dnf install poppler-utils\n"
                    f"  Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases\n"
                    f"Original error: {error_msg}"
                ) from e
            
            if (
                "tesseract" in error_lower
                and ("not found" in error_lower or "not installed" in error_lower or "cannot find" in error_lower)
            ) or "tesseractnotfounderror" in error_lower:
                raise RuntimeError(
                    f"Tesseract OCR is required but not found. "
                    f"Install it with:\n"
                    f"  macOS: brew install tesseract\n"
                    f"  Ubuntu/Debian: sudo apt-get install tesseract-ocr\n"
                    f"  Fedora/RHEL: sudo dnf install tesseract\n"
                    f"  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki\n"
                    f"After installation, ensure tesseract is in your PATH.\n"
                    f"Original error: {error_msg}"
                ) from e
            
            raise ValueError(f"Failed to extract text with OCR from PDF {pdf_path}: {error_msg}") from e

    def _extract_with_ocr_page(self, pdf_path: Path, page_number: int) -> str:
        """Extract text from a specific page using OCR."""
        try:
            logger.debug(f"Extracting text from page {page_number} of {pdf_path} using OCR")
            images = pdf2image.convert_from_path(pdf_path, first_page=page_number + 1, last_page=page_number + 1)
            if not images:
                raise IndexError(f"Could not extract page {page_number} from PDF")
            image = images[0]
            text = pytesseract.image_to_string(image, lang=self.ocr_lang)
            return text if text else ""
        except ImportError as e:
            missing_package = str(e).split()[-1] if "No module named" in str(e) else "unknown"
            raise ImportError(
                f"OCR dependencies not installed. Install with: "
                f"pip install pytesseract pillow pdf2image. "
                f"Also ensure Tesseract OCR is installed on your system. "
                f"Missing: {missing_package}"
            ) from e
        except Exception as e:
            error_msg = str(e)
            error_lower = error_msg.lower()
            
            if "poppler" in error_lower or "unable to get page count" in error_lower:
                raise RuntimeError(
                    f"Poppler is required for OCR but not found. "
                    f"Install it with:\n"
                    f"  macOS: brew install poppler\n"
                    f"  Ubuntu/Debian: sudo apt-get install poppler-utils\n"
                    f"  Fedora/RHEL: sudo dnf install poppler-utils\n"
                    f"  Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases\n"
                    f"Original error: {error_msg}"
                ) from e
            
            if (
                "tesseract" in error_lower
                and ("not found" in error_lower or "not installed" in error_lower or "cannot find" in error_lower)
            ) or "tesseractnotfounderror" in error_lower:
                raise RuntimeError(
                    f"Tesseract OCR is required but not found. "
                    f"Install it with:\n"
                    f"  macOS: brew install tesseract\n"
                    f"  Ubuntu/Debian: sudo apt-get install tesseract-ocr\n"
                    f"  Fedora/RHEL: sudo dnf install tesseract\n"
                    f"  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki\n"
                    f"After installation, ensure tesseract is in your PATH.\n"
                    f"Original error: {error_msg}"
                ) from e
            
            raise ValueError(
                f"Failed to extract text with OCR from page {page_number} of PDF {pdf_path}: {error_msg}"
            ) from e


