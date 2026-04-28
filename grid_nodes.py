import torch
import numpy as np
from PIL import Image

def make_image_grid(images, rows, cols, resize=None, background="white"):
    """
    Prepares a single grid of images. 
    """
    assert len(images) == rows * cols

    if resize is not None:
        images = [img.resize((resize, resize), Image.LANCZOS) for img in images]

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h), color=background)

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

class ImageGridComposite6:
    """
    Composites 6 images into a 2x3 grid.
    Standalone version with no external dependencies.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "columns": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "rows": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "rescale": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
            },
            "optional": {
                "image_batch": ("IMAGE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"
    CATEGORY = "Texture_Projection/Utils"

    def composite(self, columns, rows, rescale, image_batch=None, **kwargs):
        # Collect images from batch and optional inputs
        input_images = []
        
        # Add images from batch if provided
        if image_batch is not None:
            for i in range(image_batch.shape[0]):
                input_images.append(image_batch[i])
            
        # Add individual image inputs if provided
        for i in range(1, 7): # image1 to image6
            key = f"image{i}"
            if key in kwargs and kwargs[key] is not None:
                # ComfyUI image inputs are usually tensors with batch dim (B, H, W, C)
                img = kwargs[key]
                for j in range(img.shape[0]):
                    input_images.append(img[j])

        # Target count for the grid
        target_count = columns * rows
        
        if len(input_images) == 0:
            # Return a blank image if no input
            return (torch.zeros((1, rescale, rescale, 3)),)

        # Pad or trim to target count
        if len(input_images) < target_count:
            last_img = input_images[-1]
            for _ in range(target_count - len(input_images)):
                input_images.append(torch.zeros_like(last_img))
        elif len(input_images) > target_count:
            input_images = input_images[:target_count]

        # Convert torch (H, W, C) to PIL
        pil_images = []
        for img in input_images:
            # ComfyUI images are float [0, 1]
            img_np = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))

        # Make Grid using integrated utility
        grid_pil = make_image_grid(pil_images, rows=rows, cols=columns, resize=rescale)
        
        # Convert back to torch (1, H, W, C)
        grid_np = np.array(grid_pil).astype(np.float32) / 255.0
        grid_torch = torch.from_numpy(grid_np).unsqueeze(0)
        
        return (grid_torch,)

class ImageGridSplit6:
    """
    Splits a grid image back into 6 individual images.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "columns": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "rows": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image_batch", "image1", "image2", "image3", "image4", "image5", "image6")
    FUNCTION = "split"
    CATEGORY = "Texture_Projection/Utils"

    def split(self, image, columns, rows):
        # image is (1, H, W, C)
        img_torch = image[0]
        H, W, C = img_torch.shape
        
        img_h, img_w = H // rows, W // columns
        
        splitted_images = []
        for r in range(rows):
            for c in range(columns):
                # Slice the tensor
                left = c * img_w
                top = r * img_h
                right = left + img_w
                bottom = top + img_h
                
                crop = img_torch[top:bottom, left:right, :]
                # Ensure all crops are the same size (handle edge cases)
                if crop.shape[0] != img_h or crop.shape[1] != img_w:
                    # This shouldn't normally happen if grid was made correctly
                    continue
                splitted_images.append(crop)
        
        # Ensure we have at least 6 outputs for the individual pins
        outputs = [torch.stack(splitted_images)] # Index 0: Batch of all
        
        # Fill individual pins
        for i in range(6):
            if i < len(splitted_images):
                outputs.append(splitted_images[i].unsqueeze(0))
            else:
                # Pad with black if grid had fewer than 6 segments
                black = torch.zeros((1, img_h, img_w, C))
                outputs.append(black)
                
        return tuple(outputs)

NODE_CLASS_MAPPINGS = {
    "ImageGridComposite6": ImageGridComposite6,
    "ImageGridSplit6": ImageGridSplit6
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageGridComposite6": "Image Grid Composite (2x3)",
    "ImageGridSplit6": "Image Grid Split (2x3)"
}
