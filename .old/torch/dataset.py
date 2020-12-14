import os
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import one_hot_encoding as ohe
import settings


class KcaptchaDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.image_file_paths = [
            os.path.join(folder, image_file) for image_file in os.listdir(folder)
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_file_paths)

    def __getitem__(self, idx):
        image_root = self.image_file_paths[idx]
        filename = image_root.split(os.sep)[-1]
        image = Image.open(image_root)
        if self.transform is not None:
            image = self.transform(image)
        label = ohe.encode(filename.split("_")[0])
        return image, label


transform = transforms.Compose(
    [
        # transforms.ColorJitter(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)


def get_train_data_loader():
    dataset = KcaptchaDataset(settings.TRAIN_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=64, shuffle=True)


def get_test_data_loader():
    dataset = KcaptchaDataset(settings.TEST_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=True)

def get_validation_data_loader():
    dataset = KcaptchaDataset(settings.VALIDATION_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=True)