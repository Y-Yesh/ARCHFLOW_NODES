import cv2
import numpy as np
import torch
from typing import Any, Dict

class ConvertBoundedAreaToMaskNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_mask": ("IMAGE",),  # Assumes input_mask is provided as an image tensor
            }
        }

    RETURN_TYPES = ("IMAGE",)  # Specify the return type directly
    FUNCTION = "convert_bounded_area_to_mask"  # Define function name for ComfyUI
    CATEGORY = "Image Processing"  # Define the category for organization in ComfyUI

    def convert_bounded_area_to_mask(self, input_mask: torch.Tensor) -> tuple:
        """
        Convert the area bounded by a mask to a new mask.

        Parameters:
        - input_mask: A binary mask image in tensor format.

        Returns:
        - output_mask: A binary mask with the area bounded by the input mask converted to a mask.
        """

        print("Input mask shape before array conversion:", input_mask.shape)

        if isinstance(input_mask, np.ndarray):
            print ("The image is a NumPy array.")
        elif isinstance(input_mask, torch.Tensor):
            print("The image is a PyTorch tensor.")
        else:
           print("The image is neither a NumPy array nor a PyTorch tensor.") 

        # Move tensor to CPU if it's not already, and convert to a NumPy array
        input_mask = input_mask.squeeze(0).cpu().numpy()

        print("Input mask shape after array conversion:", input_mask.shape)

        if isinstance(input_mask, np.ndarray):
            print ("The image is a NumPy array.")
        elif isinstance(input_mask, torch.Tensor):
            print("The image is a PyTorch tensor.")
        else:
           print("The image is neither a NumPy array nor a PyTorch tensor.") 

        if input_mask.shape[-1] == 3:
            input_mask = cv2.cvtColor(input_mask, cv2.COLOR_RGB2GRAY)
            print("Converted input mask to grayscale.")

        print("Input mask shape after array conversion and binary conversion:", input_mask.shape)

        # Scale the input mask to 0-255 range if it’s in 0-1 range
        if input_mask.max() <= 1.0:
            input_mask = (input_mask * 255).astype(np.uint8)
        else:
            input_mask = input_mask.astype(np.uint8)

        # Ensure the input mask is in binary format
        _, binary_mask = cv2.threshold(input_mask, 127, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

        # Debugging: Check the binary mask after thresholding
        print("Max value in binary mask after thresholding:", binary_mask.max())

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Debugging: Check if any contours were found
        print("Number of contours found:", len(contours))

        '''# If no contours are found, return a blank mask
        if len(contours) == 0:
            print("No contours detected, returning blank mask.")
            output_mask = torch.zeros((1, *input_mask.shape), dtype=torch.float32)
            return (output_mask,)'''
        # Create a blank output mask
        output_mask = np.zeros_like(binary_mask)

        # Fill the bounded area defined by the contours on the output mask
        cv2.drawContours(output_mask, contours, -1, (255), thickness=cv2.FILLED)

        # Normalize the output mask to 0-1 range as expected by ComfyUI
        output_mask = output_mask.astype(np.float32) / 255.0

        # Convert back to tensor and add a channel dimension if required by ComfyUI
        output_mask = torch.from_numpy(output_mask).unsqueeze(0)  # Add channel dimension

        # Debugging: Final output mask check
        print("Output mask shape:", output_mask.shape)
        print("Max and min values in output mask:", output_mask.max(), output_mask.min())

        return (output_mask,)

# Register the node to make it available in ComfyUI
NODE_CLASS_MAPPINGS = {
    "ConvertBoundedAreaToMaskNode": ConvertBoundedAreaToMaskNode
}
