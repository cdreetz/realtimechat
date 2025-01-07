from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio
import numpy as np
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import io
import json
import uvicorn
import base64
from typing import List
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s -%(name)s -%(levelname)s -%(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.last_activity = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.last_activity[websocket] = datetime.now()

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        self.last_activity.pop(websocket, None)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

class SpeechServer:
    def __init__(self):
        self.app = FastAPI()
        self.setup_cors()
        self.setup_routes()
        self.manager = ConnectionManager()
        self.setup_models()

        # self.stt_model = "openai/whisper-large-v3-turbo" # ~1.62GB
        # self.text_model = "meta-llama/Llama-3.2-1B-Instruct" # ~2.47GB
        # self.tts_model = "microsoft/speecht5_tts"
        
    def setup_cors(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def setup_models(self):
        print("Loading models...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load models one at a time with cache clearing
        print("Loading Whisper model...")
        torch.cuda.empty_cache()
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3-turbo",
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        print("Loading Llama model...")
        torch.cuda.empty_cache()
        self.chat_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        self.chat_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        print("Loading XTTS model...")
        torch.cuda.empty_cache()
        self.tts_config = XttsConfig()
        self.tts_config.load_json("../models/xtts/config.json")  # Update with your config path
        self.tts_model = Xtts.init_from_config(self.tts_config)
        self.tts_model.load_checkpoint(self.tts_config, checkpoint_dir="../models/xtts/", eval=True)  # Update with your checkpoint path
        self.tts_model.cuda()

        # Load a reference audio file for the voice
        self.reference_audio = "path/to/reference.wav"  # Update with your reference audio path

        print("All models loaded successfully!")

    def setup_routes(self):
        @self.app.get("/")
        async def get_status():
            return {
                "status": "online",
                "device": str(self.device),
                "connections": len(self.manager.active_connections)
            }

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.manager.connect(websocket)
            try:
                while True:
                    try:
                        message = await websocket.receive_json()
                        await self.handle_websocket_message(websocket, message)
                    except WebSocketDisconnect:
                        logger.info("Client disconnected normally")
                        break
                    except Exception as e:
                        print(f"Error in websocket loop: {e}")
                        break
            finally:
                self.manager.disconect(websocket)
                logger.info("Cleaned up connection")

    async def detect_speech(self, audio_data: np.ndarray) -> bool:
        frame_length = 1024
        hop_length = 512
        frames = torch.from_numpy(audio_data.copy()).unfold(0, frame_length, hop_length)
        frames = frames.float()
        
        # Calculate energy and increase threshold
        energy = frames.pow(2).mean(dim=1)
        threshold = energy.mean() * 2.5  # Increased from 1.5 to 2.5
        
        # Add minimum energy requirement
        min_energy = 0.001  # Adjust this value based on testing
        speech_detected = (energy > threshold).any().item() and energy.mean() > min_energy
        
        if speech_detected:
            logger.info("Speech detected in audio chunk")
        return speech_detected

    async def speech_to_text(self, audio_data: np.ndarray) -> str:
        logger.info("Transcribing speech...")
        input_features = self.whisper_processor(
            audio_data, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(self.device)
        
        # Cast input to float16
        input_features = input_features.to(torch.float16)
        
        predicted_ids = self.whisper_model.generate(input_features)
        transcription = self.whisper_processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        logger.info(f"Transcribed: {transcription}")
        return transcription

    async def generate_chat_response(self, text: str) -> str:
        logger.info("Generating response...")
        inputs = self.chat_tokenizer(
            f"User: {text}\nAssistant:",
            return_tensors="pt",
            return_attention_mask=True,
            max_length=512,
            truncation=True
        ).to(self.device)
        
        outputs = self.chat_model.generate(
            **inputs,
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.chat_tokenizer.eos_token_id
        )
        
        response = self.chat_tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Assistant:")[-1].strip()
        logger.info(f"Generated response: {response}")
        return response

    async def text_to_speech(self, text: str, websocket: WebSocket) -> None:
        # Split text into sentences and chunk them
        sentences = text.split('.')
        chunks = []
        current_chunk = []
        current_length = 0
        max_chunk_length = 100
        
        for sentence in sentences:
            sentence = sentence.strip() + '.'  # Add back the period
            if not sentence:
                continue
            
            if current_length + len(sentence.split()) > max_chunk_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence.split())
            else:
                current_chunk.append(sentence)
                current_length += len(sentence.split())
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Process and stream each chunk
        for i, chunk in enumerate(chunks):
            # Generate speech using XTTS
            speech = self.tts_model.synthesize(
                chunk,
                self.tts_config,
                speaker_wav=self.reference_audio,
                gpt_cond_len=3,
                language="en"
            )
            
            # Convert to tensor and prepare for saving
            speech_tensor = torch.tensor(speech, dtype=torch.float32).unsqueeze(0)
            
            # Save this chunk to buffer
            buffer = io.BytesIO()
            torchaudio.save(
                buffer,
                speech_tensor,
                sample_rate=24000,  # XTTS uses 24kHz
                format="wav"
            )
            
            # Send this chunk to the client
            audio_b64 = base64.b64encode(buffer.getvalue()).decode()
            await websocket.send_json({
                "type": "audio_response_chunk",
                "data": audio_b64,
                "chunk": i,
                "total_chunks": len(chunks)
            })

    async def handle_websocket_message(self, websocket: WebSocket, message: dict):
        try:
            message_type = message.get("type")
            data = message.get("data")
            
            if message_type == "audio":
                # Decode base64 audio data
                audio_bytes = base64.b64decode(data)
                audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                
                # Add minimum length check
                if len(audio_array) < 2048:  # Adjust this value based on your needs
                    return
                
                # Check for speech
                if await self.detect_speech(audio_array):
                    # Convert speech to text
                    text = await self.speech_to_text(audio_array)
                    
                    # Skip empty or very short transcriptions
                    if not text or len(text.strip()) <= 1:
                        return
                        
                    await websocket.send_json({
                        "type": "transcription",
                        "data": text
                    })
                    
                    # Generate response
                    response = await self.generate_chat_response(text)
                    
                    # Add conversation context to avoid repetitive responses
                    if hasattr(self, 'last_response') and response == self.last_response:
                        return
                    self.last_response = response
                    
                    try:
                        await websocket.send_json({
                            "type": "chat_response",
                            "data": response
                        })
                    except Exception as e:
                        logger.error(f"Error sending chat response: {e}")
                    
                    # Modified TTS section
                    try:
                        await self.text_to_speech(response, websocket)
                    except Exception as e:
                        logger.error(f"Error sending audio response: {e}")
                        return
            
            elif message_type == "text":
                # Direct text input
                response = await self.generate_chat_response(data)
                await websocket.send_json({
                    "type": "chat_response",
                    "data": response
                })
                
                await self.text_to_speech(response, websocket)
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "data": f"Unknown message type: {message_type}"
                })
                
        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "data": str(e)
            })

    def run(self, host="0.0.0.0", port=8000):
        uvicorn.run(self.app, host=host, port=port)

if __name__ == "__main__":
    server = SpeechServer()
    server.run(host="0.0.0.0", port=8000)
