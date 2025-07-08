from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import io
import os
from PIL import Image
import torch
import logging
from style_transfer import StyleTransferModel
from image_utils import preprocess_image, postprocess_image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ArtifyAI API",
    description="Neural Style Transfer API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Global variables for models
style_models = {}

@app.on_event("startup")
async def startup_event():
    """Load all style transfer models on startup"""
    global style_models
    
    model_paths = {
        "starry_night": "app/static/models/starry_night.pth",
        "the_scream": "app/static/models/scream.pth", 
        "great_wave": "app/static/models/wave.pth",
        "cubism": "app/static/models/picasso.pth"
    }
    
    for style_name, model_path in model_paths.items():
        try:
            if os.path.exists(model_path):
                style_models[style_name] = StyleTransferModel(model_path)
                logger.info(f"Loaded {style_name} model")
            else:
                logger.warning(f"Model file not found: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load {style_name} model: {e}")

@app.get("/")
async def root():
    """Serve the main HTML page"""
    return FileResponse("frontend/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": len(style_models)}

@app.post("/api/style-transfer")
async def apply_style_transfer(
    file: UploadFile = File(...),
    style: str = Form(...)
):
    """Apply neural style transfer to uploaded image"""
    try:
        # Validate file
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Validate style
        if style not in style_models:
            raise HTTPException(status_code=400, detail=f"Style '{style}' not available")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Apply style transfer
        model = style_models[style]
        styled_image = model.transfer_style(processed_image)
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        styled_image.save(img_buffer, format='PNG', quality=95)
        img_buffer.seek(0)
        
        return StreamingResponse(
            img_buffer,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=artify_{style}.png"}
        )
        
    except Exception as e:
        logger.error(f"Style transfer error: {e}")
        raise HTTPException(status_code=500, detail="Style transfer failed")

@app.get("/api/styles")
async def get_available_styles():
    """Get list of available styles"""
    styles = [
        {
            "id": "starry_night",
            "name": "Starry Night",
            "description": "Van Gogh's swirling night sky",
            "artist": "Vincent van Gogh",
            "emoji": "🌟"
        },
        {
            "id": "the_scream", 
            "name": "The Scream",
            "description": "Munch's expressionist masterpiece",
            "artist": "Edvard Munch",
            "emoji": "😱"
        },
        {
            "id": "great_wave",
            "name": "Great Wave",
            "description": "Hokusai's iconic wave",
            "artist": "Katsushika Hokusai", 
            "emoji": "🌊"
        },
        {
            "id": "cubism",
            "name": "Cubism",
            "description": "Picasso's geometric abstraction",
            "artist": "Pablo Picasso",
            "emoji": "🎭"
        }
    ]
    
    # Filter only available styles
    available_styles = [s for s in styles if s["id"] in style_models]
    return {"styles": available_styles}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)