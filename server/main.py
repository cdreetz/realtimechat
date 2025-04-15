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
    SpeechT5Processor, 
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline
)
import io
import json
import uvicorn
import base64
from typing import List
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import logging
from models import build_model
from kokoro import KPipeline

from torch.serialization import add_safe_globals
import os
import requests
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
        self.app.websocket_max_message_size = 5 * 1024 * 1024  # 5MB
        self.setup_cors()
        self.setup_routes()
        self.manager = ConnectionManager()
        
        # Add model options
        self.available_language_models = {
            "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
            "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
            "phi": "microsoft/phi-4",
            "qwen": "Qwen/Qwen2-VL-2B-Instruct"
        }
        self.current_language_model = "llama1b"  # default model
        self.available_voice_models = {
            "af_heart": "Default (Bella & Sarah mix)",
            "af_bella": "Bella",
            "af_sarah": "Sarah",
            "am_adam": "Adam",
            "am_michael": "Michael",
            "bf_emma": "Emma",
            "bf_isabella": "Isabella",
            "bm_george": "George",
            "bm_lewis": "Lewis",
            "af_nicole": "Nicole",
            "af_sky": "Sky"
        }
        self.current_voice = "af_heart"  # default voice
        self.voice_dir = "voices"
        os.makedirs(self.voice_dir, exist_ok=True)
        self.setup_models()

        # self.stt_model = "openai/whisper-large-v3-turbo" # ~1.62GB
        # self.text_model = "meta-llama/Llama-3.2-1B-Instruct" # ~2.47GB
        # self.tts_model = "microsoft/speecht5_tts"
        
        # Add chat history management
        self.chat_histories = {}  # Keyed by session id
        self.session_counter = 0
        self.max_history_tokens = 2048  # Adjust based on your model's context window
        
        # System prompt to use for all conversations
        self.system_prompt = (
            "You are a helpful AI assistant. Respond naturally and directly to what "
            "the user says."
        )

    def setup_cors(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def download_voice(self, voice_name: str) -> str:
        """Get voice model path from the cloned repository"""
        voice_path = os.path.join('Kokoro-82M', 'voices', f'{voice_name}.pt')
        
        if not os.path.exists(voice_path):
            raise FileNotFoundError(f"Voice {voice_name} not found in the repository!")
        
        return voice_path

    def load_voice(self, voice_path: str) -> torch.Tensor:
        """Load voice model directly"""
        logger.info(f"Loading voice from {voice_path}")
        return torch.load(voice_path, map_location=self.device)

    def setup_models(self):
        print("Loading models...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"Using device: {self.device}")

        # Load models one at a time with cache clearing
        print("Loading Whisper model...")
        torch.cuda.empty_cache()
        self.whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "openai/whisper-large-v3-turbo",
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        
        self.whisper_processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
        
        self.whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=self.whisper_model,
            tokenizer=self.whisper_processor.tokenizer,
            feature_extractor=self.whisper_processor.feature_extractor,
            chunk_length_s=30,
            batch_size=16,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

        print(f"Loading Language model: {self.available_language_models[self.current_language_model]}...")
        torch.cuda.empty_cache()
        self.chat_tokenizer = AutoTokenizer.from_pretrained(self.available_language_models[self.current_language_model])
        self.chat_tokenizer.pad_token = self.chat_tokenizer.eos_token
        self.chat_tokenizer.padding_side = "right"
        
        self.chat_model = AutoModelForCausalLM.from_pretrained(
            self.available_language_models[self.current_language_model],
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.chat_model.config.pad_token_id = self.chat_tokenizer.pad_token_id

        print("Loading Kokoro TTS model...")
        torch.cuda.empty_cache()

        try:
            import subprocess
            result = subprocess.run(['which', 'espeak-ng'], capture_output=True, text=True)
            if not result.stdout.strip():
            print("Installing espeak-ng...")
            os.system('apt-get -qq -y install espeak-ng > /dev/null 2>&1')
        except Exception as e:
            print(f"Error checking/installing espeak-ng: {str(e)}")
        
        lang_code = self.current_voice[0]
        self.tts_pipeline = KPipeline(lang_code=lang_code, device=self.device)
        print(f"Loaded Kokoro TTS pipeline with language code: {lang_code}")
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
            session_id = str(self.session_counter)
            self.session_counter += 1

            self.chat_histories[session_id] = [
                {"role": "system", "content": self.system_prompt}
            ]

            websocket.session_id = session_id

            try:
                while True:
                    try:
                        message = await websocket.receive_json()
                        await self.handle_websocket_message(websocket, message)
                    except WebSocketDisconnect:
                        logger.info("Client disconnected normally")
                        break
                    except Exception as e:
                        logger.error(f"Error in websocket loop: {str(e)}")
                        break
            except Exception as e:
                logger.error(f"Websocket error: {str(e)}")
            finally:
                self.manager.disconnect(websocket)
                logger.info("Cleaned up connection")

        @self.app.get("/available_models")
        async def get_available_models():
            return {
                "current_model": self.current_language_model,
                "available_models": list(self.available_language_models.keys())
            }

        @self.app.post("/change_voice/{voice_name}")
        async def change_voice(voice_name: str):
            if voice_name not in self.available_voice_models:
                return {"error": f"Voice {voice_name} not available"}
            
            try:
                voice_path = self.download_voice(voice_name)
                self.voicepack = self.load_voice(voice_path)
                self.current_voice = voice_name
                
                return {
                    "status": "success",
                    "message": f"Changed voice to {voice_name}",
                    "voice_name": voice_name,
                    "voice_description": self.available_voice_models[voice_name]
                }
            except Exception as e:
                logger.error(f"Failed to change voice: {str(e)}")
                return {"error": f"Failed to change voice: {str(e)}"}

        @self.app.get("/available_voices")
        async def get_available_voices():
            return {
                "current_voice": self.current_voice,
                "available_voices": self.available_voice_models
            }

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
        result = self.whisper_pipe(
            audio_data,
            generate_kwargs={"language": "english"}
        )
        transcription = result["text"]
        logger.info(f"Transcribed: {transcription}")
        return transcription

    async def generate_chat_response(self, text: str, session_id: str) -> str:
        logger.info("Generating response...")
        
        # Initialize chat history for new connections
        #if websocket not in self.chat_histories:
        #    self.chat_histories[websocket] = [
        #        {"role": "system", "content": self.system_prompt}
        #    ]
        
        # Add user message to history
        self.chat_histories[session_id].append({"role": "user", "content": text})
        
        # Construct the full conversation history
        if "llama" in self.current_language_model:
            messages = self.chat_histories[session_id].copy()

            conversation = self.chat_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            conversation = ""
            for message in self.chat_histories[session_id]:
                role = message["role"]
                content = message["content"]
                if role == "system":
                    conversation += f"{content}\n\n"
                elif role == "user":
                    conversation += f"User: {content}\n"
                elif role == "assistant":
                    conversation += f"Assistant: {content}\n"
            
            conversation += "Assistant:"
        
        inputs = self.chat_tokenizer(conversation, return_length=True)

        # Check token count and trim history if needed
        #while inputs.length[0] > self.max_history_tokens and len(self.chat_histories[websocket]) > 2:
        #    # Always keep system prompt and remove oldest message pair
        #    self.chat_histories[session_id] = (
        #        [self.chat_histories[session_id][0]] +  # Keep system prompt
        #        self.chat_histories[session_id][3:]      # Remove oldest user+assistant pair
        #    )

        #    if "llama" in self.current_language_model:
        #        messages = self.chat_histories[session_id].copy()
        #        conversation = self.chat_tokenizer.apply_chat_template(
        #            messages,
        #            tokenize=False,
        #            add_generation_prompt=True
        #        )
        #    else:
        #        # Reconstruct conversation with trimmed history
        #        conversation = ""
        #        for message in self.chat_histories[websocket]:
        #            role = message["role"]
        #            content = message["content"]
        #            if role == "system":
        #                conversation += f"{content}\n\n"
        #            elif role == "user":
        #                conversation += f"User: {content}\n"
        #            elif role == "assistant":
        #                conversation += f"Assistant: {content}\n"
        #        conversation += "Assistant:"
        #    inputs = self.chat_tokenizer(conversation, return_length=True)
        
        # Generate response with the conversation history
        inputs = self.chat_tokenizer(
            conversation,
            return_tensors="pt",
            return_attention_mask=True,
            max_length=self.max_history_tokens,
            truncation=True,
            padding=True
        )
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.chat_model.device) for k, v in inputs.items()}
        
        outputs = self.chat_model.generate(
            **inputs,
            max_length=self.max_history_tokens,
            min_length=1,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.chat_tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
        
        response = self.chat_tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Assistant:")[-1].strip()
        
        # Add assistant response to history
        self.chat_histories[session_id].append({"role": "assistant", "content": response})
        
        logger.info(f"Generated response: {response}")
        return response

    async def text_to_speech(self, text: str, websocket: WebSocket) -> None:
        logger.info(f"Processing TTS for text: {text[:50]}...")
        
        # Add basic text normalization
        text = text.replace("...", ".").replace("..", ".")  # Fix multiple periods
        text = " ".join(text.split())  # Normalize whitespace
        
        # Split text into sentences and chunk them
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        chunks = []
        current_chunk = []
        current_length = 0
        max_chunk_length = 100
        
        for sentence in sentences:
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
        
        logger.info(f"Split into {len(chunks)} chunks")
        
        # Process and stream each chunk
        chunk_size = 256 * 1024  # 256KB chunks
        audio_chunks = []  # Store all audio chunks before sending
        
        # First generate all audio chunks
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            try:
                # Generate speech using Kokoro
                audio_generator = self.tts_pipeline(chunk, voice=self.current_voice)
                for _, _, audio in audio_generator:
                    if audio is None or len(audio) == 0:
                        logger.error("Generated audio is empty")
                        continue
                    
                    logger.info(f"Generated audio shape: {audio.shape}")
                    
                    # Convert numpy array to tensor
                    speech_tensor = audio.unsqueeze(0) if audio.dim() == 1 else audio
                    
                    # Save this chunk to buffer
                    buffer = io.BytesIO()
                    torchaudio.save(
                        buffer,
                        speech_tensor,
                        sample_rate=24000,
                        format="wav"
                    )
                    
                    audio_chunks.append(buffer.getvalue())
                
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {str(e)}")
                continue
        
        # Then send all chunks
        try:
            total_audio_chunks = len(audio_chunks)
            for i, audio_data in enumerate(audio_chunks):
                # Split into sub-chunks
                total_sub_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
                logger.info(f"Sending chunk {i+1}/{total_audio_chunks} ({total_sub_chunks} sub-chunks)")
                
                for j in range(total_sub_chunks):
                    start = j * chunk_size
                    end = min((j + 1) * chunk_size, len(audio_data))
                    audio_sub_chunk = audio_data[start:end]
                    
                    try:
                        # Send this chunk to the client
                        audio_b64 = base64.b64encode(audio_sub_chunk).decode()
                        await websocket.send_json({
                            "type": "audio_response_chunk",
                            "data": audio_b64,
                            "chunk": i,
                            "total_chunks": total_audio_chunks,
                            "sub_chunk": j,
                            "total_sub_chunks": total_sub_chunks,
                            "is_final": (i == total_audio_chunks - 1 and j == total_sub_chunks - 1)
                        })
                        logger.info(f"Sent sub-chunk {j+1}/{total_sub_chunks}")
                    except Exception as e:
                        logger.error(f"Failed to send sub-chunk: {str(e)}")
                        return  # Exit if we can't send
                    
                    # Add a small delay between chunks to prevent overwhelming the connection
                    await asyncio.sleep(0.01)
                    
        except Exception as e:
            logger.error(f"Error sending audio chunks: {str(e)}")

    async def handle_websocket_message(self, websocket: WebSocket, message: dict):
        try:
            message_type = message.get("type")
            data = message.get("data")
            session_id = websocket.session_id
            
            if message_type == "audio":
                audio_bytes = base64.b64decode(data)
                audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                
                # minimum length check
                if len(audio_array) < 2048:  
                    return
                
                if await self.detect_speech(audio_array):
                    text = await self.speech_to_text(audio_array)
                    
                    if not text or len(text.strip()) <= 1:
                        return
                        
                    await websocket.send_json({
                        "type": "transcription",
                        "data": text
                    })
                    
                    response = await self.generate_chat_response(text, websocket)
                    
                    # Add conversation context to avoid repetitive responses
                    #if hasattr(self, 'last_response') and response == self.last_response:
                    #    return
                    #self.last_response = response
                    
                    try:
                        await websocket.send_json({
                            "type": "chat_response",
                            "data": response
                        })
                    except Exception as e:
                        logger.error(f"Error sending chat response: {e}")
                    
                    logger.info("Starting TTS generation...")
                    try:
                        await self.text_to_speech(response, websocket)
                        logger.info("TTS generation completed")
                    except Exception as e:
                        logger.error(f"Error in TTS generation: {str(e)}")
                        return
            
            elif message_type == "text":
                response = await self.generate_chat_response(data, session_id)
                await websocket.send_json({
                    "type": "chat_response",
                    "data": response
                })
                
                logger.info("Starting TTS generation for text input...")
                try:
                    await self.text_to_speech(response, websocket)
                    logger.info("TTS generation completed")
                except Exception as e:
                    logger.error(f"Error in TTS generation: {str(e)}")
                    return
            
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

    def disconnect(self, websocket: WebSocket):
        # Clean up chat history when client disconnects
        if hasattr(websocket, 'session_id') and websocket.session_id in self.chat_histories:
            self.chat_histories.pop(websocket.session_id, None)
        self.active_connections.remove(websocket)
        self.last_activity.pop(websocket, None)

    def run(self, host="0.0.0.0", port=8000):
        uvicorn.run(self.app, host=host, port=port)

if __name__ == "__main__":
    server = SpeechServer()
    server.run(host="0.0.0.0", port=8000)
