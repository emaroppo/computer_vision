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

data_dir = "facial_recognition/data"


batch_size = 32
epochs = 8
workers = 8


def train(data_dir, batch_size, epochs, workers):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device: {}".format(device))
    trans = transforms.Compose(
        [np.float32, transforms.ToTensor(), fixed_image_standardization]
    )

    dataset = datasets.ImageFolder(data_dir, transform=trans)
    img_inds = np.arange(len(dataset))
    np.random.shuffle(img_inds)
    train_inds = img_inds[: int(0.8 * len(img_inds))]
    val_inds = img_inds[int(0.8 * len(img_inds)) :]

    train_loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_inds),
    )

    val_loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_inds),
    )

    resnet = InceptionResnetV1(
        classify=True, pretrained="vggface2", num_classes=len(dataset.class_to_idx)
    ).to(device)

    optimizer = optim.Adam(resnet.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer, [5, 10])

    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = {"fps": training.BatchTimer(), "acc": training.accuracy}

    writer = SummaryWriter()
    writer.iteration, writer.interval = 0, 10

    print("\n\nInitial")
    print("-" * 10)
    resnet.eval()
    training.pass_epoch(
        resnet,
        loss_fn,
        val_loader,
        batch_metrics=metrics,
        show_running=True,
        device=device,
        writer=writer,
    )

    for epoch in range(epochs):
        print("\nEpoch {}/{}".format(epoch + 1, epochs))
        print("-" * 10)

        resnet.train()
        training.pass_epoch(
            resnet,
            loss_fn,
            train_loader,
            optimizer,
            scheduler,
            batch_metrics=metrics,
            show_running=True,
            device=device,
            writer=writer,
        )

        resnet.eval()
        training.pass_epoch(
            resnet,
            loss_fn,
            val_loader,
            batch_metrics=metrics,
            show_running=True,
            device=device,
            writer=writer,
        )

    writer.close()
    return resnet


def inference_classification(
    data_dir,
    workers=8,
    model_file=None,
    classify=True,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model_file:
        resnet = torch.load(model_file).eval().to(device)
    else:
        resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    resnet = torch.load("facial_recognition/trained_resnet.pt")
    if not classify:
        resnet.classify = False
    trans = transforms.Compose(
        [np.float32, transforms.ToTensor(), fixed_image_standardization]
    )

    dataset = datasets.ImageFolder(data_dir, transform=trans)
    dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, num_workers=workers, batch_size=32)

    all_preds = []
    all_targets = []
    for x, y in loader:
        all_targets.append(y)
        x = x.to(device)

        with torch.no_grad():
            preds = resnet(x)
        all_preds.append(preds)
        all_targets.append(y)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    return all_preds
