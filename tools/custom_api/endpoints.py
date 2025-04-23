from fastapi import FastAPI, Query
from pydantic import BaseModel
import subprocess
import os

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


class SemanticTokenRequest(BaseModel):
    speaker_name: str
    text_input: str
    narration_id: str  # Unique ID for the narration

@app.post("/generate-semantic-tokens")
def generate_semantic_tokens(request: SemanticTokenRequest):
    try:
        speaker_name = request.speaker_name
        text_input = request.text_input
        narration_id = request.narration_id

        prompt_tokens = f"/opt/fish-speech/assets/{speaker_name}.npy"
        prompt_text_path = f"/opt/fish-speech/assets/{speaker_name}.txt"

        if not os.path.exists(prompt_tokens):
            return {"status": "error", "message": f"Prompt tokens not found at {prompt_tokens}"}
        if not os.path.exists(prompt_text_path):
            return {"status": "error", "message": f"Prompt text not found at {prompt_text_path}"}

        with open(prompt_text_path, "r") as f:
            prompt_text = f.read().strip()

        # Step 1: Run semantic token generation
        cmd_generate = [
            "python", "fish_speech/models/text2semantic/inference.py",
            "--text", text_input,
            "--prompt-text", prompt_text,
            "--prompt-tokens", prompt_tokens,
            "--checkpoint-path", "checkpoints/fish-speech-1.5",
            "--num-samples", "1",
            "--device", "cuda",
            "--compile",
            "--half"
        ]

        result = subprocess.run(cmd_generate, capture_output=True, text=True)
        if result.returncode != 0:
            return {
                "status": "error",
                "stage": "semantic_token_generation",
                "message": result.stderr
            }

        # Step 2: Run VQGAN inference using generated semantic tokens
        output_wav = f"/opt/fish-speech/results/speech-{narration_id}-output.wav"
        cmd_infer = [
            "python", "fish_speech/models/vqgan/inference.py",
            "-i", "temp/codes_0.npy",
            "-o", output_wav,
            "--device", "cuda",
            "--checkpoint-path", "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
        ]

        result_infer = subprocess.run(cmd_infer, capture_output=True, text=True)
        if result_infer.returncode != 0:
            return {
                "status": "error",
                "stage": "audio_inference",
                "message": result_infer.stderr
            }

        return {
            "status": "success",
            "message": "Semantic token generation and audio inference completed.",
            "output_audio_path": output_wav
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
    
# curl -X POST http://localhost:8080/generate-semantic-tokens \
#   -H "Content-Type: application/json" \
#   -d '{
#     "speaker_name": "adeline",
#     "text_input": "The global crypto market cap has also increased by 6.46 percent, reaching 2.9 trillion dollars. So, what do you guys think? Is ADA going to keep on performing, pushing through resistance and surprising everyone? Or is it just riding the hype train before a big dump? Personally, I am watching the charts closely, but I want to hear your take. Let me know what you are seeing down in the comments!",
#     "narration_id": "ada-analysis-april23"
#   }'
