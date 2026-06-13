import threading
import time
import queue
import pyttsx3
import pyaudio
import wave
import os
import ollama
import pytesseract
from mss import mss
from PIL import Image
from faster_whisper import WhisperModel

# ==========================================
# 0. System Configurations (Mac Specific)
# ==========================================
# Point pytesseract to the Homebrew installation path
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

# ==========================================
# 1. Application State & Settings

# ==========================================
LLM_MODEL_NAME = "llama3:8b"  # Ensure you ran `ollama pull llama3:8b`
CHUNK_DURATION = 5            # Listen to audio in 5-second blocks
presentation_context = []     # Master memory array

audio_queue = queue.Queue()

# ==========================================
# 2. AI Model Initialization
# ==========================================
print("Loading Local Models... This may take a moment.")

# TTS (Text-to-Speech) Offline Engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 160)

def speak(text):
    print(f"\n[Interviewer]: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

# STT (Speech-to-Text) Engine
# If it's your first time, it will download ~70MB for the tiny model
whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

# ==========================================
# 3. Screen Capture & OCR Thread
# ==========================================
def extract_screen_text():
    with mss() as sct:
        # Change monitors[1] to monitors[0] if you only have one screen
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        
        while True:
            # Grab frame and extract text
            screenshot = sct.grab(monitor)
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            text = pytesseract.image_to_string(img).strip()
            
            # Only save meaningful text blocks to avoid spamming the context
            if len(text) > 20:
                presentation_context.append(f"[SCREEN OCR]: {text}")
            
            time.sleep(10) # Scan every 10 seconds

# ==========================================
# 4. Audio Capture Thread
# ==========================================
def record_audio():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    while True:
        frames = []
        for _ in range(0, int(RATE / CHUNK * CHUNK_DURATION)):
            data = stream.read(CHUNK)
            frames.append(data)
            
        # Save temp file using thread ID to prevent overwrites
        temp_file = f"temp_{threading.get_ident()}.wav"
        wf = wave.open(temp_file, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        audio_queue.put(temp_file)

# ==========================================
# 5. Audio Transcription Thread
# ==========================================
def transcribe_audio():
    while True:
        file_path = audio_queue.get()
        segments, _ = whisper_model.transcribe(file_path, beam_size=5)
        text = "".join([segment.text for segment in segments]).strip()
        
        if text:
            presentation_context.append(f"[STUDENT SPOKE]: {text}")
            print(f"\n[Transcribed]: {text}")
        
        # Clean up the temporary audio file
        if os.path.exists(file_path):
            os.remove(file_path)

# ==========================================
# 6. AI Interviewer Logic (Ollama)
# ==========================================
def generate_question():
    while True:
        time.sleep(45) # Ask a question every 45 seconds
        
        if len(presentation_context) < 3:
            continue # Wait for more context before asking
            
        # Keep a safe slice of the most recent context
        context_str = "\n".join(presentation_context[-15:])
        
        prompt = f"""You are a software engineering technical interviewer. Based on the student's recent presentation code and speech, ask ONE short, specific follow-up question. Do not praise them. Be direct.

Context:
{context_str}"""

        try:
            response = ollama.generate(
                model=LLM_MODEL_NAME, 
                prompt=prompt,
                options={"num_ctx": 8192, "temperature": 0.7}
            )
            question = response['response'].strip()
            speak(question)
        except Exception as e:
            print(f"\n[Error generating question]: {e}")

def evaluate_performance():
    print("\n====================================================")
    print("        INITIALIZING EVALUATION PIPELINE            ")
    print("====================================================\n")
    
    # Debug Check 1: See if any data was actually captured
    total_logs = len(presentation_context)
    print(f"[DEBUG 1/4] Total context entries captured in memory: {total_logs}")
    
    if total_logs == 0:
        print("\n[CRITICAL WARNING]: No data was captured during this session!")
        return

    # Slicing context safely
    safe_context = presentation_context[-40:] if total_logs > 40 else presentation_context
    context_str = "\n".join(safe_context)
    
    print(f"[DEBUG 2/4] Packaged the most recent {len(safe_context)} entries for Ollama.")
    
    # Keep the prompt very simple now, let the schema do the heavy lifting
    prompt = f"""Review the following transcript and evaluate the student's performance out of 10.
    
Transcript:
{context_str}"""

    # 1. Define the exact JSON schema we require
    evaluation_schema = {
        "type": "object",
        "properties": {
            "technical_depth": { "type": "number" },
            "clarity_of_explanation": { "type": "number" },
            "originality": { "type": "number" },
            "understanding_of_implementation": { "type": "number" },
            "feedback_summary": { "type": "string" }
        },
        "required": [
            "technical_depth", 
            "clarity_of_explanation", 
            "originality", 
            "understanding_of_implementation", 
            "feedback_summary"
        ]
    }

    print(f"[DEBUG 3/4] Connecting to local Ollama service using model '{LLM_MODEL_NAME}'...")
    
    start_time = time.time()
    try:
        response = ollama.generate(
            model=LLM_MODEL_NAME, 
            prompt=prompt,
            # 2. Pass the schema dictionary here instead of just "json"
            format=evaluation_schema, 
            # 3. Drop temperature to 0.0 to prevent creative deviations
            options={"num_ctx": 8192, "temperature": 0.0}
        )
        
        elapsed_time = time.time() - start_time
        print(f"[DEBUG 4/4] Response received successfully in {elapsed_time:.2f} seconds!")
        
        report = response['response'].strip()
        print("\n====================================================")
        print("                FINAL EVALUATION REPORT             ")
        print("====================================================")
        print(report)
        print("====================================================\n")
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to communicate with Ollama: {e}")
# ==========================================
# 7. Main Execution Orchestrator
# ==========================================
if __name__ == "__main__":
    print("\n====================================================")
    print(" Welcome to the AI Automated Interviewer (Ollama) ")
    print("====================================================\n")
    input("Press Enter to start the presentation...")
    
    # Start all background threads as daemons (so they die when main thread dies)
    threading.Thread(target=extract_screen_text, daemon=True).start()
    threading.Thread(target=record_audio, daemon=True).start()
    threading.Thread(target=transcribe_audio, daemon=True).start()
    threading.Thread(target=generate_question, daemon=True).start()
    
    speak("I am ready. Please begin your presentation.")
    
    try:
        # Keep the main thread alive indefinitely
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Caught Ctrl+C -> Generate final report before shutting down
        print("\n\nEnding Interview and compiling data...")
        evaluate_performance()