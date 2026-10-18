from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def open_pdf(path: Any) -> Iterator[Any]:
    """Open a pypdfium2 PdfDocument with explicit close (works across pypdfium2 versions)."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(path)
    try:
        yield doc
    finally:
        doc.close()
