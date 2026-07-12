from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
import os
import logging
from typing import List, Dict, Any
from pydantic import BaseModel
from dotenv import load_dotenv
import asyncio

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s -%(name)s -%(levelname)s -%(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class TranscriptionRequest(BaseModel):
    audio_data: List[float]
    sample_rate: int = 16000

class TextRequest(BaseModel):
    text: str
    session_id: str

class InferenceServer:
    def __init__(self):
        self.app = FastAPI()
        self.setup_cors()
        self.setup_routes()
        
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

        # Add chat history management
        self.chat_histories = {}  # Keyed by session_id
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
        
        from kokoro import KPipeline
        
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
                "sessions": len(self.chat_histories)
            }

        @self.app.post("/test")
        async def test_endpoint():
            return {"status": "ok", "message": "Inference server is working"}

        @self.app.post("/transcribe")
        async def transcribe_speech(request: TranscriptionRequest):
            try:
                audio_array = np.array(request.audio_data, dtype=np.float32)
                
                # Detect speech
                if await self.detect_speech(audio_array):
                    # Convert speech to text
                    text = await self.speech_to_text(audio_array)
                    
                    # Skip empty or very short transcriptions
                    if not text or len(text.strip()) <= 1:
                        return {"text": ""}
                        
                    return {"text": text}
                else:
                    return {"text": ""}
            except Exception as e:
                logger.error(f"Error in transcription: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/generate_response")
        async def generate_response(request: TextRequest):
            try:
                response = await self.generate_chat_response(request.text, request.session_id)
                return {"response": response}
            except Exception as e:
                logger.error(f"Error in text generation: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/synthesize_speech")
        async def synthesize_speech(request: TextRequest):
            try:
                audio_data = await self.text_to_speech(request.text)
                return {"audio": audio_data.tolist()}
            except Exception as e:
                logger.error(f"Error in speech synthesis: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

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
        
        # Initialize chat history for new sessions
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = [
                {"role": "system", "content": self.system_prompt}
            ]
        
        # Add user message to history
        self.chat_histories[session_id].append({"role": "user", "content": text})
        
        # Construct the full conversation history
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
        
        # Check token count and trim history if needed
        inputs = self.chat_tokenizer(conversation, return_length=True)
        while inputs.length[0] > self.max_history_tokens and len(self.chat_histories[session_id]) > 2:
            # Always keep system prompt and remove oldest message pair
            self.chat_histories[session_id] = (
                [self.chat_histories[session_id][0]] +  # Keep system prompt
                self.chat_histories[session_id][3:]      # Remove oldest user+assistant pair
            )
            # Reconstruct conversation with trimmed history
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

    async def text_to_speech(self, text: str) -> np.ndarray:
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
        
        # Process all chunks
        all_audio = []
        
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
                    all_audio.append(audio)
                
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {str(e)}")
                continue
        
        # Concatenate all audio chunks
        if all_audio:
            full_audio = np.concatenate(all_audio)
            return full_audio
        else:
            return np.array([], dtype=np.float32)
            
    def run(self, host="0.0.0.0", port=8001):
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)

if __name__ == "__main__":
    server = InferenceServer()
    server.run(host="0.0.0.0", port=8001)
