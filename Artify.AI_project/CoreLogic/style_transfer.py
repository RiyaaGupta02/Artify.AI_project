import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class StyleTransferModel:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Denormalization for output
        self.denorm = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    
    def transfer_style(self, content_image):
        """Apply style transfer to content image"""
        # Convert to RGB if needed
        if content_image.mode != 'RGB':
            content_image = content_image.convert('RGB')
        
        # Store original size
        original_size = content_image.size
        
        # Preprocess
        input_tensor = self.transform(content_image).unsqueeze(0).to(self.device)
        
        # Apply style transfer
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Post-process
        output = output.cpu().squeeze(0)
        output = self.denorm(output)
        output = torch.clamp(output, 0, 1)
        
        # Convert back to PIL Image
        output_image = transforms.ToPILImage()(output)
        
        # Resize back to original size
        output_image = output_image.resize(original_size, Image.Resampling.LANCZOS)
        
        return output_image