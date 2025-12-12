import sys
import os
import argparse
import io
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
from torchvision import transforms
import re

# --- Path Setup ---
# Add parent directory to sys.path so we can import 'main' and 'demo' from the root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    # Import model definitions and configurations from main.py
    from main import (
        Im2LatexModel, Tokenizer, ResizeAndPad,
        CNNEncoder, ResNetEncoder, ViTEncoder, 
        AttentionDecoder, LSTMDecoder, TransformerDecoder,
        get_image_size, normalize_latex,
        DEVICE, EMBEDDING_DIM, DECODER_HIDDEN_DIM, ATTENTION_DIM, MAX_SEQ_LEN
    )
    # Import inference logic from demo.py
    from demo import predict, SOS_TOKEN, EOS_TOKEN
except ImportError:
    print("Error: Could not import from main.py or demo.py. Make sure they are in the root directory.")
    sys.exit(1)

# --- FastAPI App Setup ---
app = FastAPI()

# Allow CORS so your GitHub Pages frontend can talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your GitHub Pages URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold model and tokenizer
model = None
tokenizer = None

# --- Helper: Image Preprocessing ---
def process_image_bytes(image_bytes, encoder_type_str):
    """
    Reads image bytes, converts to Grayscale, resizes, and normalizes 
    to match the training format.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('L')
    except Exception as e:
        print(f"Error opening image: {e}")
        return None
    
    # Get size from main.py helper
    IMG_HEIGHT, IMG_WIDTH = get_image_size(encoder_type_str)
    
    transform = transforms.Compose([
        ResizeAndPad((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Add batch dimension (1, C, H, W) and move to device
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    return img_tensor

# --- Helper: Model Loader ---
def load_model_and_tokenizer(encoder_type, decoder_type, model_path, vocab_path):
    global model, tokenizer
    
    # 1. Load Tokenizer
    print(f"Loading Tokenizer from {vocab_path}...")
    tokenizer = Tokenizer(min_freq=5)
    if os.path.exists(vocab_path):
        tokenizer.load(vocab_path)
    else:
        print(f"Warning: Vocab file {vocab_path} not found. Using default vocab size.")
    
    vocab_size = tokenizer.vocab_size

    # 2. Initialize Encoder
    print(f"Initializing Model: {encoder_type} + {decoder_type}...")
    if encoder_type.lower() == 'resnet':
        encoder = ResNetEncoder(encoded_image_size=16).to(DEVICE)
        encoder_dim = 512
    elif encoder_type.lower() == 'vit':
        encoder = ViTEncoder().to(DEVICE)
        encoder_dim = 768
    else: # cnn
        encoder = CNNEncoder(encoded_image_size=16).to(DEVICE)
        encoder_dim = 512
        
    # 3. Initialize Decoder
    if decoder_type.lower() == 'attention':
        decoder = AttentionDecoder(vocab_size, EMBEDDING_DIM, DECODER_HIDDEN_DIM, encoder_dim, ATTENTION_DIM, 0).to(DEVICE)
    elif decoder_type.lower() == 'lstm':
        decoder = LSTMDecoder(vocab_size, EMBEDDING_DIM, DECODER_HIDDEN_DIM, encoder_dim, 0).to(DEVICE)
    elif decoder_type.lower() == 'transformer':
        decoder = TransformerDecoder(vocab_size, DECODER_HIDDEN_DIM, encoder_dim, 8, 4, MAX_SEQ_LEN, 0).to(DEVICE)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")

    model = Im2LatexModel(encoder, decoder).to(DEVICE)
    
    # 4. Load Weights
    if model_path and os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint, strict=False)
        else:
            model = checkpoint.to(DEVICE)
        model.eval()
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model path {model_path} not found. Using initialized weights.")

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Im2Latex Backend is running"}

@app.post("/predict")
async def predict_equation(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded. Please start server with valid arguments."}
    
    # 1. Read Image
    content = await file.read()
    
    # Determine encoder type string for resizing logic
    encoder_type_str = "cnn"
    if isinstance(model.encoder, ResNetEncoder): encoder_type_str = "resnet"
    elif isinstance(model.encoder, ViTEncoder): encoder_type_str = "vit"
    
    # 2. Preprocess
    image_tensor = process_image_bytes(content, encoder_type_str)
    if image_tensor is None:
        return {"error": "Invalid image format"}
    
    try:
        # 3. Predict (using demo.py logic)
        latex_output = predict(model, image_tensor, tokenizer)
        
        # 4. Post-processing
        latex_output = normalize_latex(latex_output)
        latex_output = re.sub(r'\s*([\[\]\(\)\{\}\^_])\s*', r'\1', latex_output) # Fix spacing
        latex_output = latex_output.replace("<space>", " ")
        latex_output = latex_output.replace(r"\bf ", "")
        latex_output = re.sub(r'\s+', ' ', latex_output).strip()
        
        return {"latex": latex_output}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# --- Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Im2Latex Backend Server")
    parser.add_argument("--encoder", type=str, default="resnet", choices=["cnn", "resnet", "vit"])
    parser.add_argument("--decoder", type=str, default="attention", choices=["lstm", "attention", "transformer"])
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--vocab_path", type=str, default="./im2latex100k/vocab.json")
    parser.add_argument("--port", type=int, default=8000)
    
    args = parser.parse_args()
    
    # Load model before starting server
    load_model_and_tokenizer(args.encoder, args.decoder, args.model_path, args.vocab_path)
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=args.port)
