from pathlib import Path
import torch, torchvision
path = Path("cifar100")
path.mkdir(exist_ok=True)
for train, name in [(True, "train.pt"), (False, "test.pt")]:
    dset = torchvision.datasets.CIFAR100(str(path), train=train, download=True)
    images = torch.tensor(dset.data)
    labels = torch.tensor(dset.targets, dtype=torch.long)
    torch.save({"images": images, "labels": labels, "classes": dset.classes}, path / name)
    counts = torch.bincount(labels, minlength=100)
    print(f"{name}: images={tuple(images.shape)} labels={tuple(labels.shape)} min_count={int(counts.min())} max_count={int(counts.max())}")
x = torch.load(path / "train.pt", map_location="cpu", weights_only=True)["images"].float().div(255)
print("mean", x.mean((0,1,2)).tolist())
print("std", x.std((0,1,2)).tolist())
