import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
import base64


@tool
def TranscribeAudioTool(file_path: str) -> str:
    """
    Returns a transcription of a audio file.

    Args:
        file_path: The path of the audio file to transcribe.
    """
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-lite', temperature=0)

    try:
        # Convert audio to base64
        with open(file_path, 'rb') as f:
            base64_file = base64.b64encode(f.read()).decode('utf-8')

        # Construct message for LLM
        message = {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'Give a precise transcription of this audio:'},
                {'type': 'file', 'source_type': 'base64', 'data': base64_file},
            ],
        }

        # Query the LLM
        response = llm.invoke([message])
        return f'Audio content:\n{response.content}'

    # Clean up the temporary file if it was created
    finally:
        # Clean up the temporary file if it was created, and it is not a test file
        if file_path and os.path.exists(file_path) and 'test' not in file_path:
            os.remove(file_path)


if __name__ == '__main__':
    source_dir = os.path.join(os.path.dirname(__file__), '..', 'resources')

    files = ['test_audio.mp3']

    for file in files:
        file_path = os.path.join(source_dir, file)
        print(TranscribeAudioTool.invoke(input={'file_path': file_path}))
