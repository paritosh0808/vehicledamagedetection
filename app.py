# app.py
import io
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw
from torchvision import transforms
from training_pipeline import get_model
import os

app = FastAPI()

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(num_classes=8)
model.load_state_dict(torch.load(os.getenv('FINAL_MODEL_PATH', 'vehicle_damage_model_final.pth'), map_location=device))
model.to(device)
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
])

CLASS_NAMES = {
    0: "minor-dent",
    1: "minor-scratch",
    2: "moderate-broken",
    3: "moderate-dent",
    4: "moderate-scratch",
    5: "severe-broken",
    6: "severe-dent",
    7: "severe-scratch"
}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # Transform image
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(input_tensor)[0]
    
    # Draw bounding boxes on image
    draw = ImageDraw.Draw(image)
    for box, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
        if score > 0.5: 
            box = box.cpu().numpy()
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1]), f"Class: {CLASS_NAMES[label.item()]}, Score: {score.item():.2f}", fill="red")
    
    # Save and return image
    output_buffer = io.BytesIO()
    image.save(output_buffer, format='JPEG')
    output_buffer.seek(0)
    
    return StreamingResponse(output_buffer, media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)