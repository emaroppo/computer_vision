from facenet_pytorch import MTCNN, training
import av
from PIL import ImageDraw
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def detect_faces_from_video(video_path):
    # Open the video file
    container = av.open(video_path)

    mtcnn = MTCNN(keep_all=True, device="cuda")
    frame_count = 0
    # Loop through each frame in the video
    for frame in container.decode(video=0):
        frame_count += 1
        # Convert to RGB
        frame = frame.to_image()
        frame = frame.convert("RGB")

        # Detect faces
        boxes, probs = mtcnn.detect(frame)

        # Draw boxes and show
        frame_draw = frame.copy()
        draw = ImageDraw.Draw(frame_draw)
        if boxes is None:
            continue
        for i, box in enumerate(boxes):
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
        frame_draw.show()

        # Get cropped and prewhitened faces
        faces = []
        for j, box in enumerate(boxes):
            faces.append(
                mtcnn(
                    frame,
                    save_path=f"results/{video_path}/frame_{frame_count}_face_{j}.png",
                )
            )
        frame.show()


def detect_faces_from_folder(folder_path, batch_size=8):
    mtcnn = MTCNN(keep_all=True, device="cuda")
    # Load the dataset
    dataset = ImageFolder(folder_path, transform=transforms.Resize((512, 512)))

    # Create a DataLoader
    loader = torch.utils.data.DataLoader(
        dataset, collate_fn=training.collate_pil, batch_size=batch_size, num_workers=4
    )
    # Iterate through the DataLoader
    for i, (x, y) in enumerate(loader):
        image_idx = i * batch_size
        print(i, x, y)
        save_paths = [
            f"facial_detection/results/{dataset.classes[k]}/{image_idx+j}.png"
            for j, k in enumerate(y)
        ]
        # replace y with "facial_detection/results/{dataset.classes[y]}/.png"
        # Get cropped and prewhitened faces
        faces = mtcnn(x, save_path=save_paths)
        # Show the prewhitened faces


detect_faces_from_folder("facial_detection/data")
