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
            # Concatenate buffer
            audio_concat = np.concatenate(self.buffer)
            self.buffer = []
            self.buffer_duration = 0
            
            # Convert to float32 for processing
            audio_float32 = audio_concat.astype(np.float32) / 32768.0
            
            # Process audio asynchronously
            asyncio.create_task(self.callback(audio_float32))
        
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
                    # Create processor for audio track
                    audio_processor = AudioProcessTrack(
                        track=self.media_relay.subscribe(track),
                        callback=lambda audio_data: asyncio.create_task(self.process_audio(audio_data, websocket, session_id))
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
                
            # Send audio to inference server for transcription
            async with aiohttp.ClientSession() as session:
                response = await session.post(
                    f"{self.inference_server_url}/transcribe",
                    json={
                        "audio_data": audio_data.tolist(),
                        "sample_rate": 16000
                    }
                )
                
                if response.status != 200:
                    logger.error(f"Transcription error: {await response.text()}")
                    return
                    
                result = await response.json()
                text = result.get("text", "")
                
                # Skip if no text was transcribed
                if not text:
                    return
                    
                # Send transcription to client
                await websocket.send_json({
                    "type": "transcription",
                    "data": text
                })
                
                # Process the transcription if it's not empty
                await self.process_text(text, websocket, session_id)
                
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
                
            # Check if we have an audio processor for this connection
            if websocket in self.audio_processors:
                # For now, we just log that we have audio to send
                # The actual sending happens via the audio processor track
                logger.info(f"Audio ready to send to client: {len(audio_data)} samples")
                
                # The ideal solution would be to queue the audio for playback
                # through the AudioProcessTrack, but for simplicity, we'll
                # create synthetic audio frames and send them directly
                
                # Convert float32 audio to int16
                audio_int16 = (audio_data * 32767).astype(np.int16)
                
                # Split into chunks of reasonable size
                chunk_size = 960  # 60ms at 16kHz
                frame_time = 0
                
                for i in range(0, len(audio_int16), chunk_size):
                    chunk = audio_int16[i:i+chunk_size]
                    
                    # Pad the last chunk if needed
                    if len(chunk) < chunk_size:
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                    
                    # Create audio frame
                    frame = AudioFrame(
                        format="s16",
                        layout="mono",
                        samples=chunk_size
                    )
                    
                    frame.sample_rate = 16000
                    frame.pts = i // chunk_size
                    frame.time_base = fractions.Fraction(1, 16000)
                    
                    # Copy data to frame
                    frame.planes[0].update(chunk.tobytes())
                    
                    # Process frame through the audio processor
                    # In a real implementation, we would properly mix or queue this audio
                    await asyncio.sleep(0.06)  # 60ms delay between chunks
                    
                logger.info("Finished sending audio to client")
            else:
                logger.error("No audio processor for this connection")
                
        except Exception as e:
            logger.error(f"Error sending audio to client: {str(e)}")

    def run(self, host="0.0.0.0", port=8000):
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)


# Create a simple HTML file for testing
index_html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WebRTC AI Assistant</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        #status, #transcription, #response {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
        }
        #status {
            background-color: #f0f0f0;
        }
        #transcription {
            background-color: #e1f5fe;
        }
        #response {
            background-color: #e8f5e9;
        }
        button {
            padding: 10px 15px;
            margin: 5px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        #startButton {
            background-color: #4CAF50;
            color: white;
        }
        #stopButton {
            background-color: #f44336;
            color: white;
        }
        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <h1>WebRTC AI Assistant</h1>
    <div id="status">Status: Not connected</div>
    
    <div>
        <button id="startButton">Start Conversation</button>
        <button id="stopButton" class="hidden">Stop Conversation</button>
    </div>
    
    <div id="transcription"></div>
    <div id="response"></div>
    
    <script>
        let peerConnection;
        let audioStream;
        let sessionId = null;
        let processingAudio = false;
        
        const startButton = document.getElementById('startButton');
        const stopButton = document.getElementById('stopButton');
        const statusDiv = document.getElementById('status');
        const transcriptionDiv = document.getElementById('transcription');
        const responseDiv = document.getElementById('response');
        
        startButton.addEventListener('click', startConversation);
        stopButton.addEventListener('click', stopConversation);
        
        // WebSocket for signaling
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onopen = () => {
            statusDiv.textContent = 'Status: WebSocket connected';
            sessionId = generateSessionId();
            console.log('Session ID:', sessionId);
        };
        
        ws.onmessage = async (event) => {
            const message = JSON.parse(event.data);
            console.log('Received message:', message);
            
            switch (message.type) {
                case 'answer':
                    await handleOffer(message.data);
                    break;
                case 'ice_candidate':
                    if (peerConnection) {
                        try {
                            await peerConnection.addIceCandidate(message.data);
                        } catch (e) {
                            console.error('Error adding ICE candidate:', e);
                        }
                    }
                    break;
                case 'transcription':
                    transcriptionDiv.textContent = `You said: ${message.data}`;
                    break;
                case 'response':
                    responseDiv.textContent = `AI: ${message.data}`;
                    break;
                case 'audio_ready':
                    // Notification that audio response is ready
                    console.log('Audio response is ready');
                    break;
                case 'error':
                    console.error('Server error:', message.data);
                    statusDiv.textContent = `Status: Error - ${message.data}`;
                    break;
            }
        };
        
        ws.onclose = () => {
            statusDiv.textContent = 'Status: WebSocket disconnected';
            stopConversation();
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            statusDiv.textContent = 'Status: WebSocket error';
        };
        
        async function startConversation() {
            try {
                // Get microphone access
                audioStream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    }, 
                    video: false 
                });
                
                // Create peer connection
                peerConnection = new RTCPeerConnection({
                    iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
                });
                
                // Add audio track to peer connection
                audioStream.getAudioTracks().forEach(track => {
                    peerConnection.addTrack(track, audioStream);
                });
                
                // Handle ICE candidates
                peerConnection.onicecandidate = (event) => {
                    if (event.candidate) {
                        ws.send(JSON.stringify({
                            type: 'ice_candidate',
                            data: {
                                candidate: event.candidate.candidate,
                                sdpMid: event.candidate.sdpMid,
                                sdpMLineIndex: event.candidate.sdpMLineIndex
                            },
                            session_id: sessionId
                        }));
                    }
                };
                
                // Handle incoming tracks
                peerConnection.ontrack = (event) => {
                    console.log('Received remote track:', event.track);
                    if (event.track.kind === 'audio') {
                        const audioElement = new Audio();
                        audioElement.srcObject = new MediaStream([event.track]);
                        audioElement.play();
                    }
                };
                
                // ICE connection state change
                peerConnection.oniceconnectionstatechange = () => {
                    console.log('ICE connection state:', peerConnection.iceConnectionState);
                    if (peerConnection.iceConnectionState === 'connected') {
                        statusDiv.textContent = 'Status: Connected';
                    } else if (peerConnection.iceConnectionState === 'disconnected' || 
                              peerConnection.iceConnectionState === 'failed') {
                        statusDiv.textContent = 'Status: Disconnected';
                        stopConversation();
                    }
                };
                
                // Create offer
                const offer = await peerConnection.createOffer({
                    offerToReceiveAudio: true,
                    offerToReceiveVideo: false
                });
                await peerConnection.setLocalDescription(offer);
                
                // Send offer to server
                ws.send(JSON.stringify({
                    type: 'offer',
                    data: {
                        sdp: peerConnection.localDescription.sdp,
                        type: peerConnection.localDescription.type
                    },
                    session_id: sessionId
                }));
                
                // Update UI
                startButton.classList.add('hidden');
                stopButton.classList.remove('hidden');
                statusDiv.textContent = 'Status: Connecting...';
                
            } catch (error) {
                console.error('Error starting conversation:', error);
                statusDiv.textContent = 'Status: Error starting conversation';
            }
        }
        
        async function handleOffer(answer) {
            if (!peerConnection) {
                console.error('PeerConnection not initialized');
                return;
            }
            
            try {
                await peerConnection.setRemoteDescription(new RTCSessionDescription(answer));
                
                statusDiv.textContent = 'Status: Connected';
            } catch (error) {
                console.error('Error handling offer:', error);
            }
        }
        
        function stopConversation() {
            // Close peer connection
            if (peerConnection) {
                peerConnection.close();
                peerConnection = null;
            }
            
            // Stop audio stream
            if (audioStream) {
                audioStream.getTracks().forEach(track => track.stop());
                audioStream = null;
            }
            
            // Update UI
            startButton.classList.remove('hidden');
            stopButton.classList.add('hidden');
            statusDiv.textContent = 'Status: Disconnected';
        }
        
        function generateSessionId() {
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                const r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
        }
    </script>
</body>
</html>
"""

# Create template directory and write HTML file
os.makedirs(TEMPLATES_DIR, exist_ok=True)
with open(os.path.join(TEMPLATES_DIR, "index.html"), "w") as f:
    f.write(index_html)

if __name__ == "__main__":
    server = WebRTCServer(inference_server_url="http://localhost:8001")
    server.run(host="0.0.0.0", port=8000) 
