from langchain_core.tools import tool
import requests
import tempfile
import os

DEFAULT_API_URL = 'https://agents-course-unit4-scoring.hf.space'


@tool
def DownloadTaskFilesTool(task_id: str) -> str:
    """
    Useful when the user tells that the question has a file attached, otherwise it should not be used.
    Downloads a document file associated with a task ID and saves it to a temporary file.
    The file will persist after the function returns and the caller is responsible for its cleanup.

    Args:
        task_id: The task ID used to download the document.

    Returns:
        The path to the temporary file where the document is saved, or an error message if download fails.
    """
    download_url = f'{DEFAULT_API_URL}/files/{task_id}'
    temp_file_path = None

    try:
        response = requests.get(download_url, stream=True, timeout=30)
        response.raise_for_status()

        # Create a temporary file to store the downloaded content.
        # delete=False ensures the file is not deleted when closed.
        # The file is created in the system's default temporary directory.
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            temp_file_path = tmp_file.name  # Get the path of the created file

        # On success, return the path to the persistent temporary file
        return f'File downloaded successfully. Path: {temp_file_path}'

    except Exception as e:
        # If a temp file was created before this error (e.g., during writing)
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return f'Error processing document for task_id {task_id}: {e}'


if __name__ == '__main__':
    question_dict = {
        'task_id': 'f918266a-b3e0-4914-865d-4faa564f1aef',
        'question': 'What is the final numeric output from the attached Python code?',
        'Level': '1',
        'file_name': 'f918266a-b3e0-4914-865d-4faa564f1aef.py',
    }

    task_to_test = question_dict['task_id']

    print(DownloadTaskFilesTool(task_to_test))
