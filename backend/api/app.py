from typing import Optional, Dict, Union
from flask import Flask, request, jsonify
import re
import os
from pathlib import Path
from dotenv import load_dotenv
import openai
import logging
from flask.wrappers import Response

WORDS_PER_MINUTE = 180

# Load environment variables from .env
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


def estimate_speech_duration(text: str) -> float:
    """
    Estimates the duration in seconds for reading the given text aloud,
    assuming an average speech rate.

    :param text: The input text.
    :return: Estimated duration in seconds.
    """
    words = len(re.findall(r'\w+', text))
    duration_minutes = words / WORDS_PER_MINUTE
    return round(duration_minutes * 60, 2)


class TextPreparerAPI:
    """
    A Flask-based API service for preparing text to fit within a given audio duration
    using OpenAI for text summarization if needed.
    """

    def __init__(self) -> None:
        self.max_audio_length_seconds = int(os.getenv('MAX_AUDIO_LENGTH_SECONDS', 8))
        openai_api_key = os.getenv('OPENAI_API_KEY')

        if not openai_api_key:
            raise ValueError("OpenAI API key is missing.")
        openai.api_key = openai_api_key

        self.app = Flask(__name__)
        self.app.logger.setLevel(logging.INFO)
        self._setup_routes()

    @staticmethod
    def _prepare_response(
            duration: float,
            duration_before: Optional[float] = None,
            updated: bool = False,
            text: Optional[str] = None
    ) -> dict:
        """
        Prepares the JSON response structure.

        :param duration: Estimated duration for the (possibly shortened) text.
        :param duration_before: Original estimated duration.
        :param updated: Whether the text was shortened.
        :param text: The text to return.
        :return: A dictionary response.
        """
        response = {
            'duration': duration,
            'updated': updated
        }
        if duration_before is not None:
            response['duration_before'] = duration_before
        if text is not None:
            response['text'] = text
        return response

    def _shorten_text_by_openai(self, text: str, max_audio_length_seconds: int) -> str:
        """
        Uses OpenAI's GPT model to shorten a text to fit within the target speech duration.

        :param text: Original text.
        :param max_audio_length_seconds: Maximum allowed duration in seconds.
        :return: Shortened text or the original text if an error occurs.
        """
        try:
            target_words = int((max_audio_length_seconds / 60) * WORDS_PER_MINUTE)

            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": (
                        "You specialize in summarizing text for spoken content.\n"
                        "- Shorten text without losing core meaning.\n"
                        "- Ensure speech sounds natural and clear.\n"
                        "- Respect word count limits to match time restrictions.\n"
                        "- Keep a conversational tone suitable for TTS systems."
                    )},
                    {"role": "user", "content": (
                        f"Please shorten the following text to fit within {max_audio_length_seconds} seconds of speech. "
                        f"Limit the result to no more than {target_words} words. "
                        f"Ensure the output is clear, concise, and retains key points:\n\n{text}"
                    )}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            self.app.logger.error(f"Error shortening text with GPT: {e}")
            return text

    def _setup_routes(self) -> None:
        """
        Registers API routes for the Flask app.
        """

        @self.app.route('/prepare_text', methods=['POST'])
        def prepare_text() -> Union[tuple[Response, int], Response]:
            """
            Endpoint that accepts JSON with a 'text' field and returns estimated duration.
            If the text is too long, it uses OpenAI to shorten it.
            """
            data = request.get_json()
            text = data.get('text')

            if not text:
                return jsonify({'message': 'The "text" field is required.'}), 422

            original_duration = estimate_speech_duration(text)

            optimized_text = text
            response_updated = False

            if original_duration > self.max_audio_length_seconds:
                optimized_text = self._shorten_text_by_openai(text, self.max_audio_length_seconds)
                optimized_duration = estimate_speech_duration(optimized_text)
                response_updated = True
            else:
                optimized_duration = original_duration

            response = self._prepare_response(
                duration=optimized_duration,
                duration_before=original_duration if response_updated else None,
                updated=response_updated,
                text=optimized_text if response_updated else None
            )

            self.app.logger.info(f"Response: {response}")
            return jsonify(response)

    def run(self, **kwargs) -> None:
        """
        Runs the Flask application.
        """
        self.app.run(**kwargs)


if __name__ == "__main__":
    api_hostname = os.getenv('API_HOST', '127.0.0.1')
    api_port = int(os.getenv('API_PORT', 8009))

    api = TextPreparerAPI()
    api.run(debug=True, host=api_hostname, port=api_port)
