# Use a base image with Python and CUDA (for GPU support) from NVIDIA's NGC registry
FROM nvcr.io/nvidia/pytorch:23.10-py3
# Changed base image to a reliable NVIDIA PyTorch image

# Set working directory
WORKDIR /app

# Install system dependencies for ffmpeg
# Ensure apt-get update is run before install
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Install openai-whisper first, then whisper-timestamped
# whisper-timestamped is an extension and needs openai-whisper
RUN pip install --upgrade pip
RUN pip install openai-whisper==20231117 # Specify a compatible version
RUN pip install whisper-timestamped==1.15.8 # Specify a compatible version
RUN pip install Flask # For our web server
RUN pip install pydub # Added for potential audio processing needs if whisper.load_audio has issues
RUN pip install runpod # NEW: Install the RunPod SDK

# Copy the worker script into the container
COPY worker.py /app/worker.py

# Expose the port our Flask app will listen on
EXPOSE 8000

# Command to run the Flask worker application
# We'll use Flask's built-in server for simplicity, but for production,
# you might consider Gunicorn or uWSGI. RunPod typically manages the process.
CMD ["python", "worker.py"]

