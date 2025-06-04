# AI Video Generator (Synthetic Characters)

This project demonstrates how to create short videos from text prompts using a diffusion-based text-to-video model. The goal is to generate synthetic characters or scenes only. Do not use this code to impersonate real people or to generate videos of real individuals.

## Setup

1. Install the required Python packages (Python 3.8+ recommended):
   ```bash
   pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
   pip install diffusers[torch] transformers
   pip install opencv-python moviepy
   ```
   Adjust the CUDA version in the PyTorch URL based on your GPU.

2. Ensure you have a machine with a GPU for efficient generation. Using CPU will be very slow.

## Usage

Run the script with a text prompt that describes the desired scene. Prompts should only reference fictional or synthetic characters.

```bash
python text_to_video_synthetic.py "a 3d animated person dancing in a futuristic city" --num_frames 16 --out_path dancing.mp4
```

This will generate about 16 frames from the prompt and compile them into `dancing.mp4`.

## Notes

- The script uses the `damo-vilab/text-to-video-ms-1.7b` model from the `diffusers` library.
- Generation speed and quality depend on your hardware.
- Avoid uploading or referencing real people's images or likenesses. This code is intended only for synthetic characters.
