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
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate
from aiortc.contrib.media import MediaRelay, MediaBlackhole, MediaRecorder
from av import AudioFrame
import fractions

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s -%(name)s -%(levelname)s -%(message)s')
logger = logging.getLogger(__name__)

# Configure paths for static files and templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR,"templates")

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

        numpy_data = frame.to_ndarray()

        if len(numpy_data.shape) > 1:
            numpy_data = numpy_data.mean(axis=1)

        if numpy_data.dtype != np.float32:
            numpy_data = numpy_data.astype(np.float32)
            if numpy_data.max() > 1.0:
                numpy_data = numpy_data / 32768.0

        self.buffer.append(numpy_data)
        self.buffer_duration += len(numpy_data) / self.sample_rate

        if self.buffer_duration >= self.target_duration:
            audio_concat = np.concatenate(self.buffer)
            self.buffer = []
            self.buffer_duration = 0

            if np.max(np.abs(audio_concat)) > 0.01:
                logger.info(f"Sending {len(audio_concat)} samples for processing")
                asyncio.create_task(self.callback(audio_concat))
        
        return frame


class WebRTCServer:
    def __init__(self, inference_server_url="http://localhost:8001"):
        self.app = FastAPI()
        self.inference_server_url = inference_server_url
        self.setup_cors()
        self.setup_routes()
        
        # Store active connections and peer connections
        self.active_connections = {}
        self.peer_connections = {}
        self.audio_processors = {}
        self.media_relay = MediaRelay()
        
        # Add template support
        self.templates = Jinja2Templates(directory=TEMPLATES_DIR)
        
        # Mount static files
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
                logger.info("Client disconnected normally")
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
                await self.disconnect(websocket)

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[websocket] = {"connected_at": datetime.now().isoformat()}
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        # Clean up WebRTC resources
        if websocket in self.peer_connections:
            pc = self.peer_connections.pop(websocket)
            await pc.close()
        
        # Remove from active connections
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
            # Create a new RTCPeerConnection
            pc = RTCPeerConnection()
            self.peer_connections[websocket] = pc

            self.active_connections[websocket]["session_id"] = session_id
            
            # Set up event handlers
            @pc.on("iceconnectionstatechange")
            async def on_iceconnectionstatechange():
                logger.info(f"ICE connection state changed to: {pc.iceConnectionState}")
                
                if pc.iceConnectionState == "failed":
                    await pc.close()
                    self.peer_connections.pop(websocket, None)

            # Handle incoming tracks (audio from client)
            @pc.on("track")
            def on_track(track):
                logger.info(f"Received track: {track.kind}")
                if track.kind == "audio":
                    relayed_track = self.media_relay.subscribe(track)

                    # Create processor for audio track
                    audio_processor = AudioProcessTrack(
                        track=relayed_track,
                        callback=lambda audio_data: asyncio.create_task(
                            self.process_audio(audio_data, websocket, session_id)
                        )
                    )
                    self.audio_processors[websocket] = audio_processor
                    
                    # Add the audio track to our response stream
                    pc.addTrack(audio_processor)
                    
                    @track.on("ended")
                    async def on_ended():
                        logger.info("Track ended")
            
            # Parse and set the remote description from client offer
            try:
                # Make sure we have a valid SDP in the offer
                if isinstance(data, dict) and "sdp" in data and "type" in data:
                    logger.info("Setting remote description from client offer")
                    await pc.setRemoteDescription(RTCSessionDescription(
                        sdp=data["sdp"],
                        type=data["type"]
                    ))
                    
                    # Create answer
                    logger.info("Creating answer")
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    
                    # Send answer back to client
                    await websocket.send_json({
                        "type": "answer",
                        "data": {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
                    })
                else:
                    logger.error(f"Invalid offer format: {data}")
                    await websocket.send_json({
                        "type": "error",
                        "data": "Invalid offer format"
                    })
            except Exception as e:
                logger.error(f"Error processing offer: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "data": f"Error processing offer: {str(e)}"
                })
                
        elif message_type == "answer":
            # Set remote description (answer)
            if websocket in self.peer_connections:
                pc = self.peer_connections[websocket]
                await pc.setRemoteDescription(RTCSessionDescription(
                    sdp=data["sdp"],
                    type=data["type"]
                ))
                
        elif message_type == "ice_candidate":
            # Add ICE candidate
            if websocket in self.peer_connections:
                pc = self.peer_connections[websocket]
                
                try:
                    if isinstance(data, dict) and "candidate" in data:
                        logger.info(f"Adding ICE candidate: {data}")
                        
                        # Extract properties from the candidate data
                        candidate = data.get("candidate", "")
                        sdpMid = data.get("sdpMid", "")
                        sdpMLineIndex = data.get("sdpMLineIndex", 0)
                        
                        # Create and add the ICE candidate
                        ice_candidate = RTCIceCandidate(
                            candidate=candidate,
                            sdpMid=sdpMid,
                            sdpMLineIndex=sdpMLineIndex
                        )
                        
                        await pc.addIceCandidate(ice_candidate)
                    else:
                        logger.error(f"Invalid ICE candidate format: {data}")
                except Exception as e:
                    logger.error(f"Error adding ICE candidate: {str(e)}")
                
        elif message_type == "text_input":
            # Handle direct text input (non-audio)
            await self.process_text(data, websocket, session_id)

    async def process_audio(self, audio_data: np.ndarray, websocket: WebSocket, session_id: str):
        """Process audio data from WebRTC, transcribe it, and generate response"""
        try:
            # First check if there's proper audio data
            if len(audio_data) == 0:
                return

            if np.max(np.abs(audio_data)) < 0.01:
                return
                
            # Send audio to inference server for transcription
            async with aiohttp.ClientSession() as session:
                logger.info("sending to inference server for transcription")
                response = await session.post(
                    f"{self.inference_server_url}/transcribe",
                    json={
                        "audio_data": audio_data.tolist(),
                        "sample_rate": 16000
                    },
                    timeout=30
                )
                
                if response.status != 200:
                    logger.error(f"Transcription error: {await response.text()}")
                    return
                    
                result = await response.json()
                text = result.get("text", "")
                
                # Skip if no text was transcribed
                if not text or len(text.strip()) <= 1:
                    return
                    
                # Send transcription to client
                await websocket.send_json({
                    "type": "transcription",
                    "data": text
                })

                response = await session.post(
                    f"{self.inference_server_url}/generate_response",
                    json={
                        "text": text,
                        "session_id": session_id
                    }
                )

                if response.status != 200:
                    logger.error(f"Text generation error: {await response.text()}")
                    return

                result = await response.json()
                response_text = result.get("responst", "")

                logger.info(f"Response: {response_text}")

                await websocket.send_json({
                    "type": "response",
                    "data": response_text
                })

                logger.info("Synthesising speech..")
                tts_response = await session.post(
                    f"{self.inference_server_url}/synthesize_speech",
                    json={
                        "text": response_text,
                        "session_id": session_id
                    }
                )

                if tts_response != 200:
                    logger.error(f"Speech synthessis error: {await tts_response.text()}")
                    return

                audio_result = await tts_response.json()
                audio_data = np.array(audio_result.get("audio", []), dtype=np.float32)

                if len(audio_data) > 0:
                    logger.info(f"Synthesized {len(audio_data)} audio samples")
                    await self.send_audio_to_client(audio_data, websocket)
                
                
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")

    async def process_text(self, text: str, websocket: WebSocket, session_id: str):
        """Process text input, generate response, and synthesize speech"""
        try:
            # Skip if text is empty
            if not text or len(text.strip()) == 0:
                return
                
            # Send to inference server for response generation
            async with aiohttp.ClientSession() as session:
                # Generate text response
                response = await session.post(
                    f"{self.inference_server_url}/generate_response",
                    json={
                        "text": text,
                        "session_id": session_id
                    }
                )
                
                if response.status != 200:
                    logger.error(f"Text generation error: {await response.text()}")
                    return
                    
                result = await response.json()
                response_text = result.get("response", "")
                
                # Send text response to client
                await websocket.send_json({
                    "type": "response",
                    "data": response_text
                })
                
                # Synthesize speech for the response
                tts_response = await session.post(
                    f"{self.inference_server_url}/synthesize_speech",
                    json={
                        "text": response_text,
                        "session_id": session_id
                    }
                )
                
                if tts_response.status != 200:
                    logger.error(f"Speech synthesis error: {await tts_response.text()}")
                    return
                    
                audio_result = await tts_response.json()
                audio_data = np.array(audio_result.get("audio", []), dtype=np.float32)
                
                if len(audio_data) > 0:
                    # Notify client that audio is ready
                    await websocket.send_json({
                        "type": "audio_ready",
                        "data": True
                    })
                    
                    # Send audio through WebRTC
                    await self.send_audio_to_client(audio_data, websocket)
                
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")

    async def send_audio_to_client(self, audio_data: np.ndarray, websocket: WebSocket):
        """Send synthesized audio back to the client via WebRTC"""
        try:
            if websocket not in self.peer_connections:
                logger.error("No peer connection for websocket")
                return

            pc = self.peer_connections[websocket]

            if not hasattr(self, '_audio_transceivers'):
                self._audio_transceivers = {}

            if websocket not in self._audio_transceivers:
                logger.info("creating new audio transceiver")

                output_track = MediaStreamTrack()
                output_track.kind = "audio"

                transceiver = pc.addTransceiver(output_track)
                self._audio_transceiver[websocket] = (transceiver, output_track)

            _, output_track = self._audio_transceivers[websocket]

            audio_int16 = (audio_data * 32767).astype(np.int16)

            chunk_size = 960
            num_chunks = (len(audio_int16) + chunk_size - 1) // chunk_size

            logger.info(f"Sending audio data in {num_chunks} chunks")

            for i in range(0, len(audio_int16), chunk_size):
                chunk = audio_int16[i:i+chunk_size]

                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

                frame = av.AudioFrame.from_ndarray(
                    chunk.reshape(1, -1),
                    format="s16",
                    layout="mono",
                )

                frames.sample_rate = 16000
                #frames.pt = i // chunk_size
                frames.time_base = fractions.Fraction(1, 16000)

                #frames.planes[0].update(chunk.tobytes())

                output_track.emit(frame)

                #self.audio_sender.track.emit(frame)

                await asyncio.sleep(0.05)

            logger.info("Finished sending audio to client")

                
        except Exception as e:
            logger.error(f"Error sending audio to client: {str(e)}")

    def run(self, host="0.0.0.0", port=8000):
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)


# Create template directory and write HTML file
#os.makedirs(TEMPLATES_DIR, exist_ok=True)
#with open(os.path.join(TEMPLATES_DIR, "index.html"), "w") as f:
#    f.write(index_html)

if __name__ == "__main__":
    server = WebRTCServer(inference_server_url="http://localhost:8001")
    server.run(host="0.0.0.0", port=8000) 
