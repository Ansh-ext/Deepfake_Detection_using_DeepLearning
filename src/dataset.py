
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import os

class DeepfakeDataset(Dataset):
    def __init__(self, root, split='train', augment=False):
        
        self.num_frames = 5 

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2),
                transforms.GaussianBlur(
                    kernel_size=3, sigma=(0.1, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])

        self.samples = []
        for label_idx, cls in enumerate(['real', 'fake']):
            cls_dir = f'{root}/{split}/{cls}'
            if not os.path.exists(cls_dir):
                continue
            for folder in sorted(os.listdir(cls_dir)):
                fpath = f'{cls_dir}/{folder}'
                if os.path.isdir(fpath) and \
                   len(os.listdir(fpath)) >= 3:
                    self.samples.append((fpath, label_idx))

        print(f'[{split:5s}] {len(self.samples):4d} sequences '
              f'| augment={augment}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder, label = self.samples[idx]
        files  = sorted(os.listdir(folder))[:self.num_frames]
        frames = []

        for fname in files:
            img = Image.open(f'{folder}/{fname}').convert('RGB')
            frames.append(self.transform(img))

        while len(frames) < self.num_frames:
            frames.append(frames[-1])
            
        return torch.stack(frames), \
               torch.tensor(label, dtype=torch.float32)
