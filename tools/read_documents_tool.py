import pymupdf
from langchain_core.tools import tool
import os


@tool
def ReadDocumentsTool(file_path: str) -> str:
    """
    Reads the context of many text based dcouments (PDF, programming code, TXT, Markdown, JSON, etc.)

    Args:
        file_path: The path of the file to read.

    Returns:
        The text content of the document, or an error message.
    """
    try:
        # Try to open the document using PyMuPDF
        # This is expected to work for PDF and DOCX files
        # This will not work if the PDF file is not text based (e.g. scanned PDF)
        with pymupdf.open(file_path) as doc:
            # This is intended to be used with small documents, so the output will be limited
            return '\n'.join([page.get_text() for page in doc])[:10000]

    except Exception:  # If PyMuPDF fails (for any reason, e.g. wrong format, encrypted)
        # Try to read the file as simple text based file
        # This is intended to work for TXT, Markdown, JSON, Programming code, etc.
        with open(file_path, 'r', encoding='utf-8') as f:
            return f'File content:\n{f.read()}'

    # Clean up the temporary file if it was created
    finally:
        # Clean up the temporary file if it was created, and it is not a test file
        if file_path and os.path.exists(file_path) and 'test' not in file_path:
            os.remove(file_path)


if __name__ == '__main__':
    source_dir = os.path.join(os.path.dirname(__file__), '..', 'resources')

    files = ['test_python.py', 'test_pdf.pdf']

    for file in files:
        file_path = os.path.join(source_dir, file)
        print(ReadDocumentsTool.invoke(input={'file_path': file_path}))
