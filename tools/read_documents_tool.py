import pymupdf
from langchain_core.tools import tool


@tool
def ReadDocumentsTool(file_path: str) -> str:
    """
    Reads the context of many text based dcouments (PDF, programming code, TXT, Markdown, JSON, etc.)

    Args:
        file_path: Path to the document file

    Returns:
        The text content of the document
    """

    try:
        # Try to open the document using PyMuPDF
        # This is expected to work for PDF and DOCX files
        # This  will not work if the PDF file is not text based (e.g. scanned PDF)
        try:
            with pymupdf.open(file_path) as doc:
                # This is intended to be used with small documents, so the output will be limited
                return "\n".join([page.get_text() for page in doc])[:10000]

        except:
            # If PyMuPDF fails, try to read the file as simple text based file
            # This is intended to work for TXT, Markdown, JSON, Programming code, etc.
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

    except Exception as e:
        return f"Error reading document: {e}"


if __name__ == "__main__":
    import os

    source_dir = os.path.join(os.path.dirname(__file__), "..", "resources")

    files = ["test_python.py", "test_pdf.pdf"]

    for file in files:
        file_path = os.path.join(source_dir, file)
        content = ReadDocumentsTool(file_path)
        print(f"Content of {file}:\n{content}\n")
