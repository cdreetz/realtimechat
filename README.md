# AI Speech Assistant

An interactive AI speech assistant that enables real-time voice conversations using state-of-the-art AI models.

## Features

- Real-time speech-to-text using OpenAI's Whisper model
- Natural language processing using Meta's Llama model 
- High-quality text-to-speech using Kokoro-82M
- WebSocket-based client-server architecture for real-time communication
- Audio streaming and chunking for smooth conversations
- Support for both voice and text input

## Requirements

- Python 3.8+
- CUDA-capable GPU recommended
- Required Python packages listed in requirements.txt

## Usage

1. Clone the repository
```
git clone https://github.com/cdreetz/realtimechat.git
cd realtimechat
```

2. Install dependencies:
```
python3 -m venv env
source env/bin/activate # on Windows use `env\Scripts\activate`
pip install -r requirements.txt
```

3. Install additional Kokoro dependencies:
```
brew install espeak-ng
git lfs install
```

4. Run the server:
```
cd server
python setup.py
python main.py
```


4. Run the client:
```
cd ../client
python main.py
```
