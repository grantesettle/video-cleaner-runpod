import os
import io
import base64
import json
import traceback
import runpod # Import the RunPod SDK
import whisper_timestamped as whisper
import numpy as np

# Load the Whisper model once when the worker starts
# Using 'large-v2' as requested, ensure your GPU has enough VRAM
print("Loading Whisper model 'large-v2'...")
model = whisper.load_model("large-v2", device="cuda") # Use "cpu" if no GPU is available or for testing

def handler(job):
    """
    This is the main handler function for RunPod serverless worker.
    It processes incoming requests (jobs) from the RunPod queue.
    """
    try:
        job_input = job['input']
        audio_base64 = job_input['audio_base64']
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_base64)
        
        # Write to a temporary file for whisper.load_audio
        temp_audio_path = "temp_audio.mp3"
        with open(temp_audio_path, "wb") as f:
            f.write(audio_bytes)

        # Load audio using whisper.load_audio
        audio = whisper.load_audio(temp_audio_path)

        # FIXED: Use whisper.transcribe (word timestamps are included by default)
        result = whisper.transcribe(model, audio, language="en")

        # Clean up temporary file
        os.remove(temp_audio_path)

        # FIX: Manually construct a clean, serializable result to ensure data integrity
        # The whisper_timestamped library uses 'text' key for word text, but main.py expects 'word'
        serializable_result = {
            "transcription": result.get("text", ""),  # Add top-level transcription
            "text": result.get("text", ""),
            "segments": [],
            "language": result.get("language", "en")
        }

        for segment in result.get("segments", []):
            new_segment = {
                "id": segment.get("id"),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment.get("text"),
                "words": []
            }
            for word_info in segment.get("words", []):
                # This is the critical fix: the library uses the 'text' key for the word.
                # We map it to the 'word' key that main.py expects.
                word_text = word_info.get('text', '').strip()
                if word_text:  # Only add non-empty words
                    new_segment["words"].append({
                        'word': word_text,  # Use 'text' key from the library's output, map to 'word'
                        'start': word_info.get('start'),
                        'end': word_info.get('end'),
                        'confidence': word_info.get('confidence', 0.0)
                    })
            serializable_result["segments"].append(new_segment)

        print(f"Processed audio segment. Found {len(serializable_result['segments'])} segments with word-level timestamps.")
        return serializable_result

    except Exception as e:
        # Log the error for debugging on RunPod logs
        print(f"Error during inference: {e}")
        traceback.print_exc()
        # Return an error dictionary as per RunPod's error handling
        return {"error": str(e)}

# Start the RunPod serverless worker
# This tells RunPod to use our 'handler' function for processing jobs.
if __name__ == '__main__':
    print("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})