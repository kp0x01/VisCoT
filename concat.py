import os
from PIL import Image

def concat_images(folder, output_path, mode="horizontal"):
    """
    Concatenate all images in a folder into a single image.

    Args:
        folder (str): Path to folder containing images.
        output_path (str): Filepath to save concatenated image.
        mode (str): "horizontal" or "vertical".
    """
    # Load all images
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            img = Image.open(os.path.join(folder, filename)).convert("RGB")
            images.append(img)

    if not images:
        raise ValueError("No images found in folder.")

    # Compute total output size
    if mode == "horizontal":
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        result = Image.new("RGB", (total_width, max_height))
        
        x_offset = 0
        for img in images:
            result.paste(img, (x_offset, 0))
            x_offset += img.width

    elif mode == "vertical":
        max_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)
        result = Image.new("RGB", (max_width, total_height))
        
        y_offset = 0
        for img in images:
            result.paste(img, (0, y_offset))
            y_offset += img.height
    else:
        raise ValueError("mode must be 'horizontal' or 'vertical'.")

    result.save(output_path)
    print(f"Saved concatenated image to {output_path}")


# Example usage:
# concat_images("path/to/your/folder", "combined.jpg", mode="horizontal")
# concat_images("path/to/your/folder", "combined_vertical.jpg", mode="vertical")

if __name__ == "__main__":
    concat_images("/workspace/data/recipeqa/2", "combined.jpg", mode="horizontal")