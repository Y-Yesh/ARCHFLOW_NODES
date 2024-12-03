from PIL import Image, ImageChops
import torch
import numpy as np

class ImageDifferenceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "Custom"

    @staticmethod
    def tensor_to_pil(tensor):
        # Convert a Tensor to a PIL image (assumes [batch, height, width, channels])
        if tensor.ndim == 4:
            tensor = tensor[0]  # Select the first image in the batch

        # Ensure tensor is in [height, width, channels] format
        if tensor.shape[-1] != 3:  # Ensure 3 channels (RGB)
            raise ValueError(f"Expected tensor with 3 channels, got {tensor.shape[-1]}")

        # Convert to a NumPy array and then to a PIL image
        image_data = (tensor.cpu().numpy() * 255).astype(np.uint8)  # Convert to 0-255 range
        return Image.fromarray(image_data, "RGB")

    @staticmethod
    def pil_to_tensor(image):
        # Convert a PIL image back to a Tensor with shape [batch, height, width, channels]
        image_data = np.array(image).astype(np.float32) / 255.0  # Normalize to 0-1 range

        # Add batch dimension and ensure channels are in the last dimension
        if image_data.ndim == 2:  # Grayscale image
            image_data = np.stack([image_data] * 3, axis=-1)  # Convert to RGB
        elif image_data.shape[-1] != 3:  # Ensure it's RGB
            raise ValueError(f"Expected image with 3 channels, got {image_data.shape[-1]}")

        # Add batch dimension [batch, height, width, channels]
        image_tensor = torch.tensor(image_data).unsqueeze(0)  # [batch, height, width, channels]
        return image_tensor


    @staticmethod
    def process(image1, image2):
        # Convert Tensors to PIL images
        image1 = ImageDifferenceNode.tensor_to_pil(image1)
        image2 = ImageDifferenceNode.tensor_to_pil(image2)

        # Resize images to match sizes
        image1 = image1.resize(image2.size)

        # Compute the difference between images
        difference_image = ImageChops.difference(image1, image2)

        # Convert the difference image back to a Tensor
        difference_tensor = ImageDifferenceNode.pil_to_tensor(difference_image)

        # Return the difference tensor
        return (difference_tensor,)


# Register the node in ComfyUI
NODE_CLASS_MAPPINGS = {
    "ImageDifference": ImageDifferenceNode
}
