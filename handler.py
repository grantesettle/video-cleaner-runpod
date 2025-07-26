import runpod
import base64
import tempfile
import os
from faster_whisper import WhisperModel

def handler(job):
    """
    Handler function for RunPod serverless endpoint
    Processes audio files and detects profanity timestamps
    """
    try:
        # Get input data
        job_input = job['input']
        audio_data = job_input.get('audio_data')
        profanity_list = job_input.get('profanity_list', [])
        language = job_input.get('language', 'en')
        
        if not audio_data:
            return {"error": "No audio_data provided"}
        
        # Decode base64 audio data
        try:
            audio_bytes = base64.b64decode(audio_data)
        except Exception as e:
            return {"error": f"Failed to decode audio data: {str(e)}"}
        
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
        
        try:
            # Load Faster Whisper model
            model = WhisperModel("base", device="cuda", compute_type="float16")
            
            # Transcribe audio with word timestamps
            segments, info = model.transcribe(
                temp_audio_path, 
                language=language,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Process segments to find profanity
            mute_segments = []
            log_entries = []
            min_duration = 0.2
            
            # Convert profanity list to lowercase for comparison
            profanity_set = set(word.lower() for word in profanity_list)
            
            for segment in segments:
                if hasattr(segment, 'words') and segment.words:
                    for word_info in segment.words:
                        word = word_info.word.strip().lower()
                        # Clean word of punctuation for comparison
                        clean_word = ''.join(c for c in word if c.isalnum())
                        
                        if clean_word in profanity_set:
                            start = word_info.start
                            end = word_info.end
                            
                            # Ensure minimum duration
                            if end - start < min_duration:
                                end = start + min_duration
                            
                            mute_segments.append([start, end])
                            log_entries.append(f"Muted '{word_info.word}' from {start:.3f}s to {end:.3f}s")
            
            # Clean up temp file
            os.unlink(temp_audio_path)
            
            # Return results in the format expected by Flask app
            return {
                "mute_segments": mute_segments,
                "log_entries": log_entries,
                "total_segments": len(mute_segments),
                "language_detected": info.language if hasattr(info, 'language') else language
            }
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            return {"error": f"Transcription failed: {str(e)}"}
            
    except Exception as e:
        return {"error": f"Handler error: {str(e)}"}

# Start the RunPod serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
