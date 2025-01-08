# Voice Chat Assistant

A real-time voice chat application that enables natural conversation with an AI assistant using speech recognition, natural language processing, and text-to-speech synthesis.

## Features

- Real-time speech recognition using Whisper
- Natural language processing using LLaMA
- High-quality text-to-speech synthesis using XTTS
- WebSocket-based communication for real-time interaction
- Support for both voice and text input
- Streaming audio response capability
- Automatic speech detection and silence detection
- Cross-platform compatibility

## System Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for optimal performance)
- Microphone for audio input
- Speakers or headphones for audio output

## Installation

1. Clone the repository:

```bash
git clone https://github.com/cdreetz/realtimechat.git
cd realtimechat
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file in the project root directory with your configuration settings
2. Set up the audio reference file path in the server configuration
3. Adjust model paths in `main.py` if needed

## Running the Application

1. Start the server on your GPU accelerated machine:

```bash
cd server
python main.py
```

2. On your smaller client device, start up the client and replace localhost with the local IP of your server:

```bash
cd client
python main.py --server http://localhost:8000
```

### Command Line Options

Client:

- `--server`: Speech server URL (default: http://localhost:8000)
- `--debug`: Enable debug logging

Server:

- Runs on port 8000 by default
- Host address can be configured in the server initialization

## Usage

1. Once both server and client are running, the system will automatically start listening for voice input
2. Speak naturally - the system will detect speech and process it
3. To send text directly, type: `text: your message`
4. Press Ctrl+C to exit

## Technical Details

### Client

- Handles audio capture and streaming
- Manages WebSocket connection to server
- Processes incoming audio responses
- Provides command-line interface

### Server

- Implements WebSocket server using FastAPI
- Manages ML models for speech processing:
  - Whisper for speech-to-text
  - LLaMA for natural language processing
  - XTTS for text-to-speech
- Handles real-time audio processing and streaming

## Troubleshooting

1. Audio Issues:

   - Check available audio devices using the debug log
   - Verify microphone permissions
   - Adjust audio input/output device settings

2. Model Loading:

   - Ensure sufficient GPU memory
   - Verify model paths and configurations
   - Check CUDA installation if using GPU

3. Connection Issues:
   - Verify server is running and accessible
   - Check WebSocket connection status
   - Ensure correct server URL in client configuration

## Performance Optimization

- The application uses CUDA when available for improved performance
- Models are loaded with float16 precision to reduce memory usage
- Audio processing includes silence detection to optimize bandwidth
- Text chunks are processed in optimal sizes for TTS

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license information here]

## Acknowledgments

- OpenAI Whisper for speech recognition
- Meta's LLaMA for natural language processing
- Coqui XTTS for text-to-speech synthesis
