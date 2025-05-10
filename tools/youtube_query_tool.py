import os
import yt_dlp
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
import base64


@tool
def YoutubeQueryTool(youtube_url: str, query: str) -> str:
    """
    Responds to a query about a Youtube video

    Args:
        youtube_url: URL of the YouTube video
        query: Question to ask about the video content
    """
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0)

    # Setup output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'youtube_output')
    os.makedirs(output_dir, exist_ok=True)

    # Change to output directory for downloads
    original_dir = os.getcwd()
    os.chdir(output_dir)

    try:
        # Download the video
        ydl_opts = {'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(youtube_url, download=True)

            # Get the most recent file (the downloaded video)
            downloaded_files = os.listdir()
            if not downloaded_files:
                raise FileNotFoundError('No files were downloaded')

            # Convert video to base64
            with open(downloaded_files[0], 'rb') as f:
                base64_file = base64.b64encode(f.read()).decode('utf-8')

        # Construct message for LLM
        message = {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': query},
                {'type': 'file', 'source_type': 'base64', 'data': base64_file},
            ],
        }

        # Query the LLM
        response = llm.invoke([message])
        return response.content

    finally:
        # Clean up downloaded files
        for file in downloaded_files:
            os.remove(file)

        # Return to original directory
        os.chdir(original_dir)
