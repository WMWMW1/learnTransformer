from torchvision import datasets, transforms
import torch

def transform():
    transform = transforms.Compose(
        
        [
        # transforms.RandomRotation(expand=False,degrees=6,),    
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.RandomErasing(0.3),
        # transforms.RandomHorizontalFlip(p=0.2),
        # transforms.RandomCrop(size=[32,32], padding=4),
        # transforms.ColorJitter(brightness=0.001, contrast=0.001, saturation=0.001, hue=0.001),  
        ]
         )
    return transform
def load_data():

    train_dataset = datasets.CIFAR10(root='./datasetForResearch', train=True, download=True, transform=transform())
    test_dataset = datasets.CIFAR10(root='./datasetForResearch', train=False, download=True, transform=transform())
   
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    return train_loader, test_loader
