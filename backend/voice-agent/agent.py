import logging
import os
from pathlib import Path
from typing import Union, AsyncIterable

import requests
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import (
    cartesia,
    openai,
    deepgram,
    noise_cancellation,
    silero,
    turn_detector,
)

# Load environment variables from .env file
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Initialize logger
logger = logging.getLogger("voice-agent")


class VoiceAgent:
    """
    VoiceAgent is responsible for setting up and running a voice assistant workflow
    using LiveKit components including STT, LLM, and TTS.
    """

    agent: cli

    def __init__(self):
        """
        Initialize the VoiceAgent by starting the LiveKit Worker CLI application.
        """
        cli.run_app(
            WorkerOptions(
                entrypoint_fnc=self.entrypoint,
                prewarm_fnc=VoiceAgent.pre_warm,
            ),
        )

    @staticmethod
    def _response_from_json(data: dict) -> dict:
        """
        Extracts relevant fields from an API JSON response.

        :param data: JSON data received from the API.
        :return: A standardized dictionary with text and duration metadata.
        """
        return {
            'text': data.get('text'),
            'duration': data.get('duration'),
            'updated': data.get('updated', False),
            'duration_before': data.get('duration_before')
        }

    @staticmethod
    def _get_api_url() -> str:
        """
        Constructs the API URL for preparing text.

        :return: A full URL string to the text preparation endpoint.
        """
        api_port = os.getenv('API_PORT')
        api_hostname = os.getenv('API_HOST')
        api_prepare_text_path = os.getenv('PREPARE_TEXT_API_PATH', 'prepare_text')
        schema = 'https' if api_port == '443' else 'http'
        return f'{schema}://{api_hostname}:{api_port}/{api_prepare_text_path}'

    def _prepare_text_by_api(self, msg: str) -> str:
        """
        Sends text to the API to prepare and potentially shorten it for speech.

        :param msg: Original input text.
        :return: Optimized text from the API or the original input if optimization fails.
        """
        api_url = self._get_api_url()
        data = {'text': msg}

        try:
            response = requests.post(api_url, json=data)

            if response.status_code == 200:
                response_json = response.json()
                response_dict = self._response_from_json(response_json)
                logger.info(f"API Response: {response_dict}")

                return response_dict.get('text', msg) if response_dict.get('updated') else msg
            else:
                logger.error(f"Error: Received status code {response.status_code} from API.")
                return msg

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return msg

    @staticmethod
    def pre_warm(proc: JobProcess) -> None:
        """
        Preloads the Voice Activity Detection model to prepare the agent.

        :param proc: The JobProcess object to store preloaded models.
        """
        proc.userdata["vad"] = silero.VAD.load()

    async def before_tts_cb(
            self,
            agent: VoicePipelineAgent,
            text_gen: Union[str, AsyncIterable[str]]
    ) -> str:
        """
        Callback to process text before sending it to the Text-to-Speech engine.

        :param agent: The VoicePipelineAgent handling the interaction.
        :param text_gen: Either a full string or an asynchronous iterable yielding text chunks.
        :return: Fully prepared text suitable for TTS.
        """
        if isinstance(text_gen, str):
            return text_gen

        text = ""
        try:
            async for chunk in text_gen:
                text += chunk
        except Exception as e:
            logger.error(f"Error while reading text stream: {e}")

        logger.debug(f"Text received from LLM: '{text}'")
        if text:
            text = self._prepare_text_by_api(text)
            logger.debug(f"Text sent to TTS: '{text}'")

        return text

    async def entrypoint(self, ctx: JobContext) -> None:
        """
        The main entry point for handling new voice interactions.

        :param ctx: JobContext object representing the voice session.
        """
        initial_ctx = llm.ChatContext().append(
            role="system",
            text=(
                "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
                "You should use short and concise responses, avoiding unpronounceable punctuation. "
                "You were created as a demo to showcase the capabilities of LiveKit's agents framework."
            ),
        )

        logger.info(f"Connecting to room {ctx.room.name}")
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

        participant = await ctx.wait_for_participant()
        logger.info(f"Starting voice assistant for participant {participant.identity}")

        agent = VoicePipelineAgent(
            vad=ctx.proc.userdata["vad"],
            stt=deepgram.STT(),
            llm=openai.LLM(model="gpt-4o-mini"),
            before_tts_cb=self.before_tts_cb,
            tts=cartesia.TTS(),
            turn_detector=turn_detector.EOUModel(),
            min_endpointing_delay=0.5,
            max_endpointing_delay=5.0,
            noise_cancellation=noise_cancellation.BVC(),
            chat_ctx=initial_ctx,
        )

        usage_collector = metrics.UsageCollector()

        @agent.on("metrics_collected")
        def on_metrics_collected(agent_metrics: metrics.AgentMetrics) -> None:
            metrics.log_metrics(agent_metrics)
            usage_collector.collect(agent_metrics)

        agent.start(ctx.room, participant)
        await agent.say("Hey, how can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    VoiceAgent()
