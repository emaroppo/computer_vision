from facenet_pytorch import MTCNN, training
import av
from PIL import ImageDraw
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch


class FacialDetection:
    def __init__(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = MTCNN(keep_all=True, device=self.device)

    def from_video(
        self, video_path, output_path="computer_vision/facial_detection/results"
    ):
        # Open the video file
        container = av.open(video_path)

        frame_count = 0
        # Loop through each frame in the video
        for frame in container.decode(video=0):
            frame_count += 1
            # Convert to RGB
            frame = frame.to_image()
            frame = frame.convert("RGB")

            # Detect faces
            # maybe add option to return probs?
            boxes, probs = self.model.detect(frame)

            # Draw boxes and show
            frame_draw = frame.copy()
            draw = ImageDraw.Draw(frame_draw)
            faces = []
            if boxes is None:
                continue
            for i, box in enumerate(boxes):
                draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

                # Get cropped faces
                faces.append(
                    self.model(
                        frame,
                        save_path=f"{output_path}/{os.path.basename(video_path)}/frame_{frame_count}_face_{i}.png",
                    )
                )
            frame_draw.show()
            frame.show()

    def from_folder(self, folder_path, batch_size=32):
        # Load the dataset
        dataset = ImageFolder(folder_path, transform=transforms.Resize((512, 512)))

        # Create a DataLoader
        loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=training.collate_pil,
            batch_size=batch_size,
            num_workers=4,
        )
        # Iterate through the DataLoader
        all_faces = []
        for i, (x, y) in enumerate(loader):
            image_idx = i * batch_size
            save_paths = [
                f"computer_vision/facial_detection/results/{dataset.classes[k]}/{image_idx+j}.png"
                for j, k in enumerate(y)
            ]
            faces = self.model(x, save_path=save_paths)
            all_faces.extend(faces)
        return all_faces
