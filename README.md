# Voice assistant that validates audio duration

A voice assistant that shortens lengthy responses to ensure the ideal audio duration.

## Features

- Voice interface based on LiveKit
- GPT-4 for response generation
- Automatic shortening of long responses
- Cartesia for text-to-speech conversion
- Precise audio duration control
- Web interface on Next.js

## Project Structure

```
project/
├── backend                     # Backend on Python
│          ├── __init__.py
│          ├── api              # Flask app API
│          │   ├── __init__.py
│          │   └── app.py       # Main API server
│          └── voice-agent      # LiveKit agent
│              ├── __init__.py
│              └── agent.py     # Voice agent logic
├── frontend                    # Frontend on Next.js
│          ├── app              # Next.js pages
│          ├── components
│          ├── hooks
├── requirements.txt            # Dependencies: LiveKit, OpenAI, Deepgram,Flask ...
└── venv                        # Virtual environment
└── .env
└── .env.example
```

## Installation

### Requirements

- Python 3.10+
- Node.js 23+
- OpenAI API key
- LiveKit API keys
- Deepgram API keys
- Cartesia API keys

### Initialize Backend
```bash
python3 -m venv venv  
source venv/bin/activate  
pip install -r requirements.txt  
```
Copy a `.env` file from `.env.example` and set variables.

## Configuration

### Maximum Audio Duration

Set the maximum duration in seconds in the API's `.env` file:
```env
MAX_AUDIO_LENGTH_SECONDS=5
```

### Initialize Frontend
```bash
npm --prefix frontend install  
```

The API server utilizes Flask and OpenAI to validate audio duration.
## Start API
```bash
python3 backend/api/app.py  
```

The Voice Agent utilizes LiveKit for voice interactions and comes with its own set of dependencies.
#### Start Voice Agent
```bash
python3 backend/voice-agent/main.py dev
```

#### Start Frontend
```bash
npm --prefix frontend run dev
```

## Here’s how it works, in different words

1. The user interacts with the assistant through the web interface.
2. The voice agent converts spoken language into text and generates a response using GPT.
3. Before converting the response into speech, its duration is checked via the API server.
4. If the response is too long, it is automatically shortened while maintaining the core meaning.
5. The shortened text is then converted into speech and played back to the user.
