from PIL import Image, ImageOps
import io

def preprocess_image(image, max_size=1024):
    """Preprocess image for style transfer"""
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize if too large
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Auto-orient based on EXIF data
    try:
        image = ImageOps.exif_transpose(image)
    except:
        pass  # Skip if no EXIF data
    
    return image

def postprocess_image(image):
    """Post-process styled image"""
    # Ensure RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image

def validate_image(file_data):
    """Validate uploaded image file"""
    try:
        image = Image.open(io.BytesIO(file_data))
        image.verify()
        return True
    except:
        return False

def get_image_info(image):
    """Get image information"""
    return {
        "size": image.size,
        "mode": image.mode,
        "format": image.format
    }