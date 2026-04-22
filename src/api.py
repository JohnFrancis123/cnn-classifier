from io import BytesIO

import torch
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from src.model import SimpleCNN

app = FastAPI(title="CNN Classifier API")
model: SimpleCNN | None = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Prefer CUDA for inference when available.
CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

transform = transforms.Compose([ #Match the training-time preprocessing pipeline.
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def _load_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]: #Support raw state dicts and full training checkpoints.
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError: #Fallback for older torch versions.
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint did not contain a valid model state dict")
    return state_dict


def setup_model(checkpoint: str) -> None: #Load weights once and reuse the model across requests.
    global model
    model = SimpleCNN(num_classes=10).to(device)
    model.load_state_dict(_load_state_dict(checkpoint))
    model.eval()


@app.post("/predict")
async def predict(file: UploadFile) -> dict[str, str | float]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Image required")

    try: #Keep decode and inference failures behind one client-facing error.
        img = Image.open(BytesIO(await file.read())).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
        return {
            "prediction": CLASSES[logits.argmax(1).item()],
            "confidence": round(probs.max().item(), 4),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc