from fastapi import FastAPI, UploadFile, File
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import io

app = FastAPI(title="BLIP Image Captioning API")

# Load model once at startup
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = AutoModelForVision2Seq.from_pretrained("Salesforce/blip-image-captioning-large")

@app.post("/generate_caption/")
async def generate_caption(file: UploadFile = File(...)):
    # Read image file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Generate caption
    inputs = processor(images=image, return_tensors="pt")
    output_ids = model.generate(**inputs, max_length=50)
    caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    return {"caption": caption}
