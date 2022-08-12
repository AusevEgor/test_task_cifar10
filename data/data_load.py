import torch
import torchvision
from torchvision.transforms import transforms


transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
def download_CIFAR10(batch_size):
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms)
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


    # test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
def download_CIFAR100(batch_size):
    test_data = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transforms)
    train_data = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

# download_CIFAR100(32)

