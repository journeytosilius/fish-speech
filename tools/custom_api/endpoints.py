from fastapi import FastAPI, Query
from pydantic import BaseModel
import subprocess

app = FastAPI()

class InferenceRequest(BaseModel):
    audio_input_path: str
    fake_file_path: str

#this endpoint generates the LORA for a speaker. Point it to a wav file with rich audio, of a minute duration at least and generate the .npy file
@app.post("/run-vqgan")
def run_vqgan(request: InferenceRequest):
    try:
        # Construct the command
        cmd = [
            "python", "fish_speech/models/vqgan/inference.py",
            "-i", request.audio_input_path,
            "--checkpoint-path", "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
        ]

        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check success
        if result.returncode != 0:
            return {
                "status": "error",
                "message": result.stderr
            }

        # Move the resulting fake.npy to desired location
        mv_cmd = ["mv", "fake.npy", request.fake_file_path]
        subprocess.run(mv_cmd, check=True)

        return {
            "status": "success",
            "message": "VQGAN inference completed and output moved.",
            "output_path": request.fake_file_path
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
# curl -X POST http://localhost:8080/run-vqgan \
#   -H "Content-Type: application/json" \
#   -d '{
#     "audio_input_path": "/opt/fish-speech/assets/adeline_speech_full.wav",
#     "fake_file_path": "/opt/fish-speech/assets/adeline.npy"
# }'
