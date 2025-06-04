import argparse
import torch
import numpy as np
from diffusers import DiffusionPipeline
from moviepy.editor import ImageSequenceClip


def main(prompt: str, num_frames: int = 16, out_path: str = "output.mp4", fps: int = 8) -> None:
    """Generate a short video from a text prompt using a diffusion model.

    Parameters
    ----------
    prompt : str
        Text describing the desired scene with fictional or synthetic characters.
    num_frames : int, optional
        Number of frames to generate. Default is 16.
    out_path : str, optional
        Output video file path. Default is "output.mp4".
    fps : int, optional
        Frames per second for the output video. Default is 8.
    """
    model_id = "damo-vilab/text-to-video-ms-1.7b"
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    result = pipe(prompt, num_frames=num_frames)
    frames = result.frames

    frame_arrays = [np.array(frame) for frame in frames]
    clip = ImageSequenceClip(frame_arrays, fps=fps)
    clip.write_videofile(out_path, codec="libx264")
    print(f"Video saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video of synthetic characters from text.")
    parser.add_argument("prompt", help="Text describing the scene (fictional characters only)")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to generate")
    parser.add_argument("--out_path", default="output.mp4", help="Output video file path")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for the output video")
    args = parser.parse_args()

    main(args.prompt, args.num_frames, args.out_path, args.fps)
