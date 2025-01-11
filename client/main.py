import asyncio
import websockets
import sounddevice as sd
import numpy as np
import json
import base64
import soundfile as sf
import io
import click
from queue import Queue
from threading import Thread
import logging
from datetime import datetime, timedelta
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, sample_rate=16000, channels=1, dtype=np.float32):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.audio_queue = Queue()
        self.is_recording = False
        self.is_hearing_audio = False
        self.is_playing = False  # Add flag to track when we're playing audio
        
        logger.info("Available audio devices:")
        logger.info(sd.query_devices())
        
    def audio_callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Only process input when not playing audio
        if not self.is_playing:
            audio_level = np.max(np.abs(indata))
            if audio_level > 0.05:
                if not self.is_hearing_audio:
                    logger.info("Audio started...")
                    self.is_hearing_audio = True
            elif self.is_hearing_audio:
                logger.info("Audio ended...")
                self.is_hearing_audio = False
            
            self.audio_queue.put(indata.copy())

    def start_recording(self):
        self.is_recording = True
        
        device_info = sd.query_devices(1, 'input')
        logger.info(f"Using input device: {device_info['name']}")  # Only log device name
        
        # Force 16kHz sample rate for speech recognition
        self.stream = sd.InputStream(
            device=1,
            channels=self.channels,
            samplerate=16000,  # Force 16kHz
            dtype=self.dtype,
            callback=self.audio_callback,
            blocksize=1024
        )
        self.stream.start()
        logger.info("Listening...")

    def stop_recording(self):
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()

    def get_audio_chunk(self, timeout=0.1):
        try:
            return self.audio_queue.get(timeout=timeout)
        except:
            return None

class SpeechClient:
    def __init__(self, server_url):
        self.server_url = server_url.replace('http', 'ws') + '/ws'
        self.audio_processor = AudioProcessor()
        self.websocket = None
        self.running = False
        self.last_audio_time = None  # Track when we last detected significant audio
        self.silence_threshold = 1.0  # Wait for 1 second of silence before sending
        self._connection_lock = asyncio.Lock()
        
    async def ensure_connection(self):
        async with self._connection_lock:
            try:
                if self.websocket is None:
                    self.websocket = await websockets.connect(self.server_url)
                    logger.info("Connected to speech server")
                await self.websocket.ping()
                return True
            except Exception as e:
                logger.error(f"Failed to connect to server: {e}")
                self.websocket = None
                return False

    async def send_message(self, message):
        if not await self.ensure_connection():
            return False
        
        try:
            await self.websocket.send(json.dumps(message))
            return True
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket error while sending: {e}")
            self.websocket = None
            return False
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False

    async def send_audio(self, audio_data):
        # Ensure audio data is in float32 format and normalized
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Ensure audio is normalized between -1 and 1
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        message = {
            "type": "audio",
            "data": base64.b64encode(audio_data.tobytes()).decode(),
            "sample_rate": self.audio_processor.sample_rate,
            "channels": self.audio_processor.channels
        }
        success = await self.send_message(message)
        if success:
            logger.debug("Audio chunk sent successfully")
        return success

    async def send_text(self, text):
        message = {
            "type": "text",
            "data": text
        }
        return await self.send_message(message)

    async def handle_server_messages(self):
        audio_chunks = {}
        
        while self.running:
            try:
                if not await self.ensure_connection():
                    await asyncio.sleep(1)
                    continue

                message = await self.websocket.recv()
                data = json.loads(message)
                
                if data["type"] == "transcription":
                    logger.info(f"Transcription: {data['data']}")
                
                elif data["type"] == "chat_response":
                    logger.info(f"Assistant: {data['data']}")
                
                elif data["type"] == "audio_response_chunk":
                    # Create storage for this chunk if it doesn't exist
                    chunk_id = data["chunk"]
                    if chunk_id not in audio_chunks:
                        audio_chunks[chunk_id] = [None] * data["total_sub_chunks"]
                    
                    # Store this sub-chunk
                    audio_chunks[chunk_id][data["sub_chunk"]] = data["data"]
                    
                    # Check if we have all sub-chunks for this chunk
                    if None not in audio_chunks[chunk_id]:
                        try:
                            # Set playing flag before playing audio
                            self.audio_processor.is_playing = True
                            
                            # Combine all sub-chunks
                            full_chunk = b''.join([base64.b64decode(chunk) for chunk in audio_chunks[chunk_id]])
                            
                            # Convert to audio and play
                            audio_data = io.BytesIO(full_chunk)
                            audio_array, samplerate = sf.read(audio_data)
                            sd.play(audio_array, samplerate)
                            sd.wait()  # Wait for this chunk to finish playing
                            
                            # Add a small delay after playback
                            await asyncio.sleep(0.1)
                            
                        finally:
                            # Reset playing flag after audio is done
                            self.audio_processor.is_playing = False
                        
                        # Clean up if this was the final chunk
                        if data.get("is_final", False):
                            audio_chunks.clear()
                        else:
                            # Remove this chunk's data to free memory
                            del audio_chunks[chunk_id]
                
                elif data["type"] == "error":
                    logger.error(f"Server error: {data['data']}")

            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                self.websocket = None
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error handling server message: {e}")
                await asyncio.sleep(1)

    async def process_audio(self):
        chunks = []
        total_frames = 0
        self.last_audio_time = None
        
        while self.running:
            try:
                chunk = self.audio_processor.get_audio_chunk()
                
                if chunk is not None:
                    # Check audio level
                    audio_level = np.max(np.abs(chunk))
                    current_time = time.time()
                    
                    # Update last_audio_time if we detect significant audio
                    if audio_level > 0.01:  # Adjust this threshold as needed
                        self.last_audio_time = current_time
                    
                    # Amplify the audio signal
                    chunk = chunk * 5.0  # Increase volume
                    chunks.append(chunk)
                    total_frames += len(chunk)
                    
                    # Only send audio if we have enough data and silence
                    if (total_frames >= self.audio_processor.sample_rate * 0.5 and 
                        (self.last_audio_time is None or 
                         current_time - self.last_audio_time > self.silence_threshold)):
                        
                        if chunks:  # Make sure we have audio to send
                            audio_data = np.concatenate(chunks)
                            logger.debug(f"Sending audio chunk: shape={audio_data.shape}")  # Move to debug level
                            await self.send_audio(audio_data)
                            chunks = []
                            total_frames = 0
                            self.last_audio_time = None  # Reset the timer
                
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                await asyncio.sleep(0.1)

    async def start(self):
        self.running = True
        
        # Start recording audio
        self.audio_processor.start_recording()

        # Create tasks for processing audio and handling messages
        tasks = [
            asyncio.create_task(self.process_audio()),
            asyncio.create_task(self.handle_server_messages())
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Tasks cancelled")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await self.stop()

    async def stop(self):
        self.running = False
        self.audio_processor.stop_recording()
        if self.websocket is not None:
            try:
                await self.websocket.close()
            except:
                pass
        self.websocket = None

class CLI:
    def __init__(self, server_url):
        self.client = SpeechClient(server_url)

    async def handle_input(self):
        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(None, input)
                if line.startswith('text:'):
                    text = line[5:].strip()
                    await self.client.send_text(text)
            except EOFError:
                break
            except Exception as e:
                logger.error(f"Error handling input: {e}")

    async def run(self):
        print("Starting interactive mode...")
        print("Press Ctrl+C to exit")
        print("Type 'text: your message' to send text directly")

        try:
            input_task = asyncio.create_task(self.handle_input())
            client_task = asyncio.create_task(self.client.start())
            
            await asyncio.gather(input_task, client_task)
        except KeyboardInterrupt:
            print("\nShutting down...")
        except Exception as e:
            logger.error(f"Error in CLI: {e}")
        finally:
            await self.client.stop()

@click.command()
@click.option('--server', default='http://localhost:8000', 
              help='Speech server URL (default: http://localhost:8000)')
@click.option('--debug/--no-debug', default=False,
              help='Enable debug logging')
def main(server, debug):
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    cli = CLI(server)
    asyncio.run(cli.run())

if __name__ == "__main__":
    main()
