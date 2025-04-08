from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import logging
import asyncio
import aiohttp
import os
import uuid
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime

# For WebRTC
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRelay, MediaBlackhole, MediaRecorder
from av import AudioFrame
import fractions

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s -%(name)s -%(levelname)s -%(message)s')
logger = logging.getLogger(__name__)

# Configure paths for static files and templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Ensure directories exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)



# Custom audio track for processing audio from WebRTC
class AudioProcessTrack(MediaStreamTrack):
    kind = "audio"
    
    def __init__(self, track, callback):
        super().__init__()
        self.track = track
        self.callback = callback
        self.buffer = []
        self.buffer_duration = 0
        self.target_duration = 1.0  # Process 1 second of audio at a time
        self.sample_rate = 16000  # Target sample rate
        
    async def recv(self):
        frame = await self.track.recv()
        
        # Convert to mono if needed and update sample rate
        if frame.format.name != "s16" or frame.layout.name != "mono" or frame.sample_rate != self.sample_rate:
            # Convert to mono s16 PCM at 16kHz
            frame = frame.reformat(
                format="s16",
                layout="mono",
                sample_rate=self.sample_rate
            )
        
        # Process audio data
        audio_data = frame.to_ndarray()
        self.buffer.append(audio_data)
        self.buffer_duration += frame.samples / frame.sample_rate
        
        if self.buffer_duration >= self.target_duration:
            audio_concat = np.concatenate(self.buffer)
            self.buffer = []
            self.buffer_duration = 0
            
            audio_float32 = audio_concat.astype(np.float32) / 32768.0
            
            asyncio.create_task(self.callback(audio_float32))
        
        return frame


class WebRTCServer:
    def __init__(self, inference_server_url="http://localhost:8001"):
        self.app = FastAPI()
        self.inference_server_url = inference_server_url
        self.setup_cors()
        self.setup_routes()
        
        # active connections
        self.active_connections = {}
        self.peer_connections = {}
        self.audio_processors = {}
        self.media_relay = MediaRelay()
        
        # template support
        self.templates = Jinja2Templates(directory=TEMPLATES_DIR)
        
        self.app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def setup_cors(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def setup_routes(self):
        @self.app.get("/", response_class=HTMLResponse)
        async def get_index(request: Request):
            return self.templates.TemplateResponse("index.html", {"request": request})

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.connect(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    await self.handle_websocket_message(websocket, message)
            except WebSocketDisconnect:
                await self.disconnect(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
                await self.disconnect(websocket)

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[websocket] = {"connected_at": datetime.now().isoformat()}
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        if websocket in self.peer_connections:
            pc = self.peer_connections.pop(websocket)
            await pc.close()
        
        if websocket in self.active_connections:
            self.active_connections.pop(websocket)
        
        logger.info(f"Client disconnected. Remaining connections: {len(self.active_connections)}")

    async def handle_websocket_message(self, websocket: WebSocket, message: dict):
        message_type = message.get("type")
        data = message.get("data")
        session_id = message.get("session_id")
        
        if not session_id:
            session_id = str(uuid.uuid4())
        
        self.active_connections[websocket]["session_id"] = session_id
        
        if message_type == "offer":
            pc = RTCPeerConnection()
            self.peer_connections[websocket] = pc
            
            @pc.on("iceconnectionstatechange")
            async def on_iceconnectionstatechange():
                logger.info(f"ICE connection state changed to:
