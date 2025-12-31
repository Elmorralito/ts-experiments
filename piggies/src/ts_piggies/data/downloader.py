"""File downloader module for downloading files from URLs."""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import re
from pathlib import Path
import time
from typing import Any, List, Literal, Self, Sequence
from urllib.parse import urlparse, unquote
import warnings

import requests
from requests.exceptions import RequestException, Timeout, ConnectionError as RequestsConnectionError
from .model import URL   

logger = logging.getLogger(__name__)


class Downloader(ABC):
    """Abstract base class for downloaders."""

    @abstractmethod
    def download(self, url: str, *, overwrite: Literal["ALL", "FILE", "DIRECTORY", "NONE"] | bool = True) -> Self:
        """Download an object from a URL."""

    @property
    @abstractmethod
    def result(self) -> Any:
        """Get the result of the download."""


class FileDownloader(Downloader):
    """Downloads files from URLs with error handling and progress tracking."""

    __slots__ = ("output_path", "timeout", "chunk_size", "verify_ssl")

    def __init__(
        self,
        output_path: str | Path | None = None,
        timeout: int = 30,
        chunk_size: int = 8192,
        verify_ssl: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the FileDownloader.

        Args:
            output_path: Path where the file should be saved. Can be a file path or directory.
                         If directory, filename will be extracted from URL during download.
                         If not provided, uses the current working directory.
            timeout: Request timeout in seconds. Defaults to 30.
            chunk_size: Size of chunks to read when downloading. Defaults to 8192 bytes.
            verify_ssl: Whether to verify SSL certificates. Defaults to True.
        """
        logger.info(f"Initializing FileDownloader with timeout={timeout}, chunk_size={chunk_size}, verify_ssl={verify_ssl}")
        if output_path is None:
            output_path = Path.cwd()
        else:
            output_path = output_path if isinstance(output_path, Path) else Path(str(output_path))

        self.output_path = output_path
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.verify_ssl = verify_ssl or False

    def get_filename_from_url(
        self,
        url: str | URL,
        check_headers: bool = True,
        default: str = "downloaded_file",
    ) -> str:
        """
        Extract filename from a URL, optionally checking HTTP headers.
        Tries all URLs in the URL model (primary and fallback) until a filename is found.

        Args:
            url: The URL to extract the filename from. Can be a string or URL model.
            check_headers: Whether to make a HEAD request to check Content-Disposition header.
                          Defaults to True.
            default: Default filename if none can be determined. Defaults to "downloaded_file".

        Returns:
            The extracted filename.

        Raises:
            ValueError: If URL is invalid.
        """
        if not url or not isinstance(url, (str, URL)):
            raise ValueError("URL must be a non-empty string or URL model")

        if isinstance(url, str):
            url = URL(url=url)

        filename = None

        # Try each URL in the urls property (primary, then fallback)
        for url_str in url.urls:
            if not url_str or not url_str.strip():
                continue

            parsed_url = urlparse(url_str)
            logger.debug(f"Parsed URL: {parsed_url}")
            if not parsed_url.scheme or not parsed_url.netloc:
                logger.warning(f"Invalid URL format, skipping: {url_str}")
                continue

            if check_headers:
                try:
                    logger.debug(f"Checking headers for URL: {url_str}")
                    response = requests.head(
                        url_str,
                        timeout=self.timeout,
                        verify=self.verify_ssl,
                        allow_redirects=True,
                    )
                    content_disposition = response.headers.get("Content-Disposition", "")
                    logger.debug(f"Content-Disposition: {content_disposition}")
                    if content_disposition:
                        filename_match = re.search(
                            r'filename[*]?=["\']?([^"\';]+)["\']?',
                            content_disposition,
                            re.IGNORECASE,
                        )
                        if filename_match:
                            filename = unquote(filename_match.group(1))
                            break
                except (RequestException, Timeout, RequestsConnectionError) as e:
                    logger.debug(f"Failed to check headers for {url_str}: {e}")
                    continue

            if not filename:
                path = parsed_url.path
                if path:
                    candidate_filename = os.path.basename(path)
                    if candidate_filename and candidate_filename != "/":
                        filename = unquote(candidate_filename)
                        break

        if not filename or filename == "/":
            filename = default

        return filename

    def download(self, url: str | URL, *, overwrite: Literal["ALL", "FILE", "DIRECTORY", "NONE"] | bool = True) -> Self:
        """
        Download a file from a URL. Tries primary URL first, then fallback if available.

        Args:
            url: The URL of the file to download. Can be a string or URL model with fallback.
            overwrite: Overwrite behavior. Can be "ALL", "FILE", "DIRECTORY", "NONE", or bool.
                      Defaults to True (converted to "ALL").

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If URL is invalid or output path is invalid.
            FileExistsError: If file exists and overwrite doesn't allow it.
            requests.RequestException: If download fails from all URLs.
        """
        if overwrite is True:
            overwrite = "ALL"
        elif overwrite is False:
            overwrite = "NONE"

        if not url or not isinstance(url, (str, URL)):
            raise ValueError("URL must be a non-empty string or URL model")

        if isinstance(url, str):
            url = URL(url=url)

        output_path = self.output_path.resolve()
        
        # If output_path is a directory or doesn't exist and parent is a directory, use filename from URL
        if output_path.is_dir() or (not output_path.exists() and output_path.parent.is_dir()):
            filename = self.get_filename_from_url(url)
            output_path = output_path if output_path.is_dir() else output_path.parent
            output_path = output_path / filename
        elif not output_path.exists() and not output_path.parent.exists():
            # If parent doesn't exist, create it
            output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and overwrite not in ["ALL", "FILE"]:
            raise FileExistsError(f"File already exists at {output_path}. Use overwrite='ALL' or overwrite='FILE' to replace it.")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Try each URL in the urls property (primary, then fallback)
        last_error = None
        for url_str in url.urls:
            if not url_str or not url_str.strip():
                continue
                
            parsed_url = urlparse(url_str)
            if not parsed_url.scheme or not parsed_url.netloc:
                logger.warning(f"Invalid URL format, skipping: {url_str}")
                continue

            try:
                logger.debug(f"Downloading file from {url_str} to {output_path}")
                response = requests.get(
                    url_str,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                    stream=True,
                )
                response.raise_for_status()
                logger.debug(f"Response status: {response.status_code}")
                with open(output_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            file.write(chunk)
                logger.info(f"Successfully downloaded {url_str} to {output_path}")
                self.output_path = output_path
                return self
            except Timeout as e:
                last_error = RequestException(f"Request timed out after {self.timeout} seconds for {url_str}: {str(e)}")
                logger.warning(f"Timeout downloading from {url_str}, trying next URL if available")
                continue
            except RequestsConnectionError as e:
                last_error = RequestException(f"Connection error while downloading from {url_str}: {str(e)}")
                logger.exception(f"Connection error downloading from {url_str}, trying next URL if available")
                continue
            except RequestException as e:
                last_error = RequestException(f"Failed to download file from {url_str}: {str(e)}")
                logger.exception(f"Failed to download from {url_str}, trying next URL if available")
                continue
            except OSError as e:
                raise OSError(f"Failed to write file to {output_path}: {str(e)}") from e

        # If we get here, all URLs failed
        if last_error:
            raise last_error

        raise ValueError(f"No valid URLs found in URL model: {url}")

    @property
    def result(self) -> List[Path]:
        """Get the result of the download."""
        return [self.output_path]


class DirectoryDownloader(FileDownloader):
    """Downloads files from URLs to a specific directory."""

    __slots__ = ("temp_path", "num_threads", "sleep_time")

    def __init__(
        self,
        output_path: str | Path | None = None,
        timeout: int = 30,
        chunk_size: int = 8192,
        verify_ssl: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the DirectoryDownloader.

        Args:
            output_path: Path to the directory where the files should be saved.
                         If not provided, uses the current working directory.
            timeout: Request timeout in seconds. Defaults to 30.
            chunk_size: Size of chunks to read when downloading. Defaults to 8192 bytes.
            verify_ssl: Whether to verify SSL certificates. Defaults to True.
            num_threads: Number of threads to use for downloading. Defaults to 1.
            sleep_time: Time to sleep between downloads. Defaults to 0 seconds.
        """
        super().__init__(output_path, timeout, chunk_size, verify_ssl, **kwargs)
        self.temp_path = self.output_path
        self.num_threads = kwargs.get("num_threads") or 1
        self.sleep_time = kwargs.get("sleep_time") or 0

    def download_to_directory(
        self,   
        url: str | URL,
        overwrite: Literal["ALL", "FILE", "DIRECTORY", "NONE"] | bool = True,
    ) -> Path:
        """
        Download a file from a URL to a specific directory. Tries primary URL first, then fallback.

        Args:
            url: The URL of the file to download. Can be a string or URL model with fallback.
            overwrite: Overwrite behavior. Can be "ALL", "FILE", "DIRECTORY", "NONE", or bool.
                      Defaults to True (converted to "ALL").

        Returns:
            Path to the downloaded file.

        Raises:
            ValueError: If URL or directory is invalid.
            FileExistsError: If file exists and overwrite doesn't allow it.
            requests.RequestException: If download fails from all URLs.
        """
        # Normalize overwrite parameter
        if overwrite is True:
            overwrite = "ALL"
        elif overwrite is False:
            overwrite = "NONE"

        if not url or not isinstance(url, (str, URL)):
            raise ValueError("URL must be a non-empty string or URL model")

        output_path = self.output_path.resolve()
        if output_path.is_dir():
            filename = self.get_filename_from_url(url)
            output_path = output_path / filename

        logger.debug(f"Output path: {output_path}")
        if output_path.exists() and overwrite not in ["ALL", "FILE"]:
            raise FileExistsError(
                f"File already exists at {output_path}. Use overwrite='ALL' or overwrite='FILE' to replace it."
            )
        elif output_path.exists() and overwrite in ["ALL", "FILE"]:
            # Only unlink if it's a file, not a directory
            if output_path.is_file():
                output_path.unlink(missing_ok=True)
            else:
                logger.warning(f"Cannot unlink {output_path} - it is not a file")
        if output_path.exists() and output_path.is_file() and overwrite in [False, "NONE"]:
            logger.warning(f"File already exists at {output_path} and not allowed to overwrite. Skipping download.")
            return output_path

        original_output_path = self.output_path
        self.output_path = output_path
        try:
            super().download(url, overwrite=overwrite)
        finally:
            self.output_path = original_output_path
            return output_path

    def download(
        self,
        url: str | URL,
        *urls: Sequence[str | URL],
        overwrite: Literal["ALL", "FILE", "DIRECTORY", "NONE"] | bool = True,
    ) -> Self:
        """
        Download one or more files from URLs to a directory. Each URL can be a string or URL model with fallback.

        If a download fails (including when all fallback URLs fail), execution continues with the next URL
        in the batch instead of raising an exception. Failed downloads are logged as warnings.

        Args:
            url: The first URL of the file to download. Can be a string or URL model.
            *urls: Additional URLs to download. Each can be a string or URL model.
            overwrite: Overwrite behavior. Can be "ALL", "FILE", "DIRECTORY", "NONE", or bool.
                      Defaults to True (converted to "ALL").

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If URLs are invalid or directory is invalid.
        """
        # Normalize overwrite parameter
        if overwrite is True:
            overwrite = "ALL"
        elif overwrite is False:
            overwrite = "NONE"

        all_urls = [url] + list(urls)
        
        if not all_urls:
            raise ValueError("At least one URL must be provided")
        
        for url_item in all_urls:
            if not url_item or not isinstance(url_item, (str, URL)):
                raise ValueError("All URLs must be non-empty strings or URL models")

        self.manage_directory(overwrite=overwrite)
        total_urls = len(all_urls)
        logger.debug(f"Downloading {total_urls} file(s) to {self.output_path}")
        
        failed_downloads = []
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [
                executor.submit(self.download_to_directory, url_item, overwrite=overwrite)
                for url_item in all_urls
            ]
            for i, future in enumerate(futures, 1):
                try:
                    future.result()
                    if self.sleep_time > 0 and i < total_urls:
                        logger.debug(f"Using {self.sleep_time} seconds of sleep before next download. URL: {all_urls[i-1]}")
                        time.sleep(self.sleep_time)
                except Exception as e:
                    if isinstance(all_urls[i-1], URL):
                        fallback_str = f", fallbacks: {', '.join(all_urls[i-1].fallback_urls)}" if all_urls[i-1].fallback_urls else ""
                        url_repr = f"{all_urls[i-1].url}{fallback_str}"
                    else:
                        url_repr = str(all_urls[i-1])
                    logger.exception(f"Failed to download {url_repr}: {e}. Continuing with next URL.")
                    failed_downloads.append((all_urls[i-1], e))
                    continue

        if failed_downloads:
            logger.warning(f"Completed download batch with {len(failed_downloads)} failed download(s) out of {total_urls} total")
            for url_item, error in failed_downloads:
                if isinstance(url_item, URL):
                    fallback_str = f", fallbacks: {', '.join(url_item.fallback_urls)}" if url_item.fallback_urls else ""
                    url_repr = f"{url_item.url}{fallback_str}"
                else:
                    url_repr = str(url_item)
                logger.debug(f"  - Failed: {url_repr}: {error}")

        return self

    def manage_directory(
        self, 
        overwrite: Literal["ALL", "FILE", "DIRECTORY", "NONE"] | bool = True
    ) -> Path:
        """
        Manage the output directory, creating it if needed and optionally clearing it.

        Args:
            overwrite: Overwrite behavior. Can be "ALL", "FILE", "DIRECTORY", "NONE", or bool.
                      - "ALL" or "DIRECTORY": Clears existing directory contents
                      - "FILE" or "NONE" or other: Does not clear directory, only allows file-level overwrites
                      Defaults to True (converted to "ALL").

        Returns:
            Path to the managed directory.

        Raises:
            FileExistsError: If directory path is a file or symlink.
        """
        # Normalize overwrite parameter
        if overwrite is True:
            overwrite = "ALL"
        elif overwrite is False:
            overwrite = "NONE"

        output_path = self.output_path.resolve()
        if output_path.is_file() or output_path.is_symlink():
            raise FileExistsError(f"Directory {output_path} is a file or is a symlink.")
        
        # Only clear directory when overwrite is "ALL" or "DIRECTORY"
        # When overwrite is "FILE", "NONE", or anything else, only allow file-level overwrites
        if output_path.exists() and overwrite in ["ALL", "DIRECTORY"]:
            logger.debug(f"Overwriting directory {output_path} (clearing all contents)")
            for file in output_path.iterdir():
                file.unlink()
            output_path.rmdir()
        elif output_path.exists() and overwrite not in ["ALL", "DIRECTORY"]:
            logger.debug(f"Directory {output_path} exists, will only overwrite individual files if allowed")

        output_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory {output_path} created/verified")
        return output_path

    @property
    def result(self) -> List[Path]:
        """Get the result of the download."""
        if not self.temp_path.exists():
            return []
        if self.temp_path.is_dir():
            return sorted([path for path in self.temp_path.iterdir() if path.is_file()])
        return [self.temp_path]
