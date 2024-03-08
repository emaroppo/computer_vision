from facenet_pytorch import (
    InceptionResnetV1,
    fixed_image_standardization,
    training,
)
import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
from datetime import datetime

# TO DO: handle dataset for training and inference separately


class FacialRecognition:
    def __init__(
        self,
        data_dir="computer_vision/facial_recognition/data",
        workers=8,
        model_file_path=None,
    ) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.workers = workers

        self.data_dir = data_dir

        # Convert to tensor and normalize
        self.preprocess = transforms.Compose(
            [np.float32, transforms.ToTensor(), fixed_image_standardization]
        )
        self.dataset = datasets.ImageFolder(
            self.data_dir,
            transform=self.preprocess,
        )

        # Load model from file or instantiate new model
        if model_file_path:
            self.model = torch.load(model_file_path).to(self.device)
        else:
            self.model = InceptionResnetV1(
                classify=True,
                pretrained="vggface2",
                num_classes=len(self.dataset.class_to_idx),
            ).to(self.device)

    def split_data(self, batch_size=32):
        img_inds = np.arange(len(self.dataset))
        np.random.shuffle(img_inds)
        train_inds = img_inds[: int(0.8 * len(img_inds))]
        val_inds = img_inds[int(0.8 * len(img_inds)) :]

        self.train_loader = DataLoader(
            self.dataset,
            num_workers=self.workers,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(train_inds),
        )

        self.val_loader = DataLoader(
            self.dataset,
            num_workers=self.workers,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(val_inds),
        )

    def train(self, epochs=8, save_model=True):

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Decay LR by a factor of 0.1 every 5 epochs
        scheduler = MultiStepLR(optimizer, [5, 10])

        # Loss function for multi-class classification
        loss_fn = torch.nn.CrossEntropyLoss()

        metrics = {"fps": training.BatchTimer(), "acc": training.accuracy}
        writer = SummaryWriter()
        writer.iteration, writer.interval = 0, 10

        print("\n\nInitial")
        print("-" * 10)
        self.model.eval()
        training.pass_epoch(
            self.model,
            loss_fn,
            self.val_loader,
            batch_metrics=metrics,
            show_running=True,
            device=self.device,
            writer=writer,
        )

        for epoch in range(epochs):
            print("\nEpoch {}/{}".format(epoch + 1, epochs))
            print("-" * 10)

            self.model.train()
            training.pass_epoch(
                self.model,
                loss_fn,
                self.train_loader,
                optimizer,
                scheduler,
                batch_metrics=metrics,
                show_running=True,
                device=self.device,
                writer=writer,
            )

            self.model.eval()
            training.pass_epoch(
                self.model,
                loss_fn,
                self.val_loader,
                batch_metrics=metrics,
                show_running=True,
                device=self.device,
                writer=writer,
            )

        writer.close()
        if save_model:
            torch.save(
                self.model,
                f"computer_vision/facial_recognition/models/trained_resnet_{datetime.now()}.pt",
            )
        # maybe update self.model? add option via argument?
        return self.model

    def predict(
        self,
        classify=True,
    ):

        if not classify:
            self.model.classify = False

        self.dataset.idx_to_class = {i: c for c, i in self.dataset.class_to_idx.items()}
        loader = DataLoader(self.dataset, num_workers=self.workers, batch_size=32)

        all_preds = []
        for x, _ in loader:
            x = x.to(self.device)
            with torch.no_grad():
                preds = self.model(x)
            all_preds.append(preds)

        all_preds = torch.cat(all_preds)

        if classify:
            all_preds = all_preds.argmax(dim=1)
            all_labels = [self.dataset.idx_to_class[i] for i in all_preds]
            return all_preds, all_labels

        all_labels = [self.dataset.idx_to_class[i] for i in all_preds]
        return all_preds, all_labels
