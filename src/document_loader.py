import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Union

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".html", ".htm"}


def load_text(path: Union[Path, str]) -> str:
    """
    Loads raw text from a given file path based on its file extension.
    Supports PDF (.pdf), Plain Text (.txt), Markdown (.md), and HTML (.html, .htm).

    Args:
        path: Path object or string representing file path.

    Returns:
        Clean plain text string extracted from the document.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is unsupported or parsing fails.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path_obj}")

    suffix = path_obj.suffix.lower()

    if suffix == ".pdf":
        if not HAS_PYPDF:
            raise ImportError("pypdf is required to process PDF files. Please install pypdf.")
        reader = PdfReader(path_obj)
        pages_text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages_text.append(t)
        return "\n".join(pages_text)

    elif suffix in (".txt", ".md"):
        return path_obj.read_text(encoding="utf-8", errors="ignore")

    elif suffix in (".html", ".htm"):
        content = path_obj.read_text(encoding="utf-8", errors="ignore")
        if HAS_BS4:
            soup = BeautifulSoup(content, "html.parser")
            for element in soup(["script", "style"]):
                element.decompose()
            text = soup.get_text(separator=" ")
            return " ".join(text.split())
        else:
            import re
            text = re.sub(r"<[^>]+>", " ", content)
            return " ".join(text.split())

    else:
        raise ValueError(f"unsupported: {suffix}")


def load_document(path: Union[Path, str]) -> Dict[str, Union[str, int]]:
    """
    Loads a single document file, extracts its plain text, and returns metadata.

    Args:
        path: Path to the document.

    Returns:
        Dict containing source, path, format, text, char_count, sample.
    """
    path_obj = Path(path)
    text = load_text(path_obj)
    sample = text[:60].replace("\n", " ")
    return {
        "source": path_obj.name,
        "path": str(path_obj),
        "format": path_obj.suffix.lower(),
        "text": text,
        "char_count": len(text),
        "sample": sample,
    }


def load_directory(
    dir_path: Union[Path, str],
    recursive: bool = True
) -> List[Dict[str, Union[str, int]]]:
    """
    Recursively iterates over a directory to load all supported documents into plain text form.
    Gracefully skips unreadable, corrupt, missing, or unsupported files without killing execution.

    Args:
        dir_path: Path to target directory.
        recursive: Whether to scan subdirectories (default True).

    Returns:
        List of loaded document dictionaries.
    """
    target_dir = Path(dir_path)
    if not target_dir.exists() or not target_dir.is_dir():
        logging.warning("Directory does not exist or is not a directory: %s", target_dir)
        return []

    docs = []
    pattern = "**/*" if recursive else "*"
    paths = sorted(target_dir.glob(pattern))

    for path in paths:
        if not path.is_file():
            continue
        try:
            doc = load_document(path)
            docs.append(doc)
            print(f"OK {path.name}: {doc['char_count']} chars | {doc['sample']!r}")
        except Exception as e:
            print(f"SKIP {path.name}: {e}")

    return docs


def main():
    print("\n=======================================================")
    print(" [DOCUMENT INTAKE] MULTI-FORMAT DOCUMENT PROCESSING")
    print("=======================================================\n")


    data_dir = Path("data")
    if not data_dir.exists():
        logging.error("Data directory 'data' not found.")
        return

    print(f"Scanning directory: {data_dir.resolve()}\n")
    docs = load_directory(data_dir)

    print(f"\n-------------------------------------------------------")
    print(f" Intake Summary: Successfully loaded {len(docs)} document(s)")
    print(f"-------------------------------------------------------")
    total_chars = sum(d["char_count"] for d in docs)
    print(f" Total Character Count: {total_chars:,} chars\n")

    for idx, doc in enumerate(docs, 1):
        print(f"[{idx}] Source: {doc['source']} ({doc['format']}) | {doc['char_count']} chars")
        print(f"    Sample: {doc['sample']!r}")
        print("-" * 55)


if __name__ == "__main__":
    main()
