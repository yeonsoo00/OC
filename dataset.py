import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import traceback

class PatchDataset(Dataset):
    def __init__(self, root_dir, patch_size=128, transform=None, max_patches_per_image=12, max_annotations_per_image=100):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.transform = transform
        self.max_patches_per_image = max_patches_per_image
        self.max_annotations_per_image = max_annotations_per_image
        self.image_paths = []
        self.annotation_paths = []
        self._load_image_paths()

    def _load_image_paths(self):
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('_A.jpg'):
                    self.image_paths.append(os.path.join(subdir, file))
                    annotation_file = file.replace('jpg', 'txt')
                    self.annotation_paths.append(os.path.join(subdir, annotation_file))
        print(f"Loaded {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            annotation_path = self.annotation_paths[idx]

            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
            if not os.path.exists(annotation_path):
                raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

            image = Image.open(img_path).convert('RGB')
            patches = self._extract_patches(image)

            if self.transform:
                patches = [self.transform(patch) for patch in patches]

            # Pad or trim the patches list to ensure a uniform number of patches
            if len(patches) < self.max_patches_per_image:
                padding = self.max_patches_per_image - len(patches)
                patches += [torch.zeros(3, self.patch_size, self.patch_size) for _ in range(padding)]
            else:
                patches = patches[:self.max_patches_per_image]

            patches = torch.stack(patches)

            # Read YOLO annotations
            annotations = self._load_annotations(annotation_path)

            # Pad or trim annotations to ensure uniform shape
            if len(annotations) < self.max_annotations_per_image:
                padding = self.max_annotations_per_image - len(annotations)
                annotations = torch.cat([annotations, torch.zeros(padding, 5)], dim=0)
            else:
                annotations = annotations[:self.max_annotations_per_image]

            return patches, annotations

        except Exception as e:
            print(f"Error loading data for index {idx}: {e}")
            traceback.print_exc()
            raise e

    def _extract_patches(self, image):
        width, height = image.size
        patches = []

        for i in range(0, width, self.patch_size):
            for j in range(0, height, self.patch_size):
                box = (i, j, i + self.patch_size, j + self.patch_size)
                patch = image.crop(box)
                if patch.size == (self.patch_size, self.patch_size):
                    patches.append(patch)

        return patches

    def _load_annotations(self, annotation_path):
        annotations = []
        with open(annotation_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                class_label = int(parts[0])
                bbox = list(map(float, parts[1:]))
                annotations.append([class_label] + bbox)
        annotations = torch.tensor(annotations, dtype=torch.float32)
        return annotations

def create_dataloader(root_dir, batch_size=32, patch_size=128, max_patches_per_image=12, max_annotations_per_image=100, num_workers=0, shuffle=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = PatchDataset(root_dir=root_dir, patch_size=patch_size, transform=transform, max_patches_per_image=max_patches_per_image, max_annotations_per_image=max_annotations_per_image)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    return dataloader

# Example usage
root_dir = 'Data/Annotations'
dataloader = create_dataloader(root_dir, batch_size=32)

for patches, annotations in dataloader:
    print(patches.shape)
    print(annotations.shape)
