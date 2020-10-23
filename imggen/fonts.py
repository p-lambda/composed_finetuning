from pathlib import Path

import h5py
import numpy as np
from torchvision.datasets.vision import VisionDataset
from PIL import Image

import requests
import zipfile
from tqdm import tqdm

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

class Fonts(VisionDataset):

    url_id = '0B0GtwTQ6IF9AU3NOdzFzUWZ0aDQ'
    base_folder = 'fonts'

    def __init__(self, root, split='train',
                 transform=None, target_transform=None, download=True,
                 denoise=False, denoise_transform=None, num_fonts_pi=None,
                 num_examples=2500):
        '''
        Args:
            root (str): path
            num_train_domains (int): number of train domains up to 41443
            test_mean_chars (bool): Use the mean characters as test set
            split (str): 'train', 'val', 'test'
            transform: input transformation
            target_transform: target transformation
            download (bool): download or not
        '''
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.denoise = denoise
        self.denoise_transform = denoise_transform

        self.path = Path(self.root) / self.base_folder
        self.path.mkdir(parents=True, exist_ok=True)
        self.download_path = self.path / 'fonts.hdf5'

        if download:
            self.download()

        with h5py.File(str(self.download_path), 'r') as f:
            data_by_domain = f['fonts'][()]

        np.random.seed(484347)
        # limit the number of fonts
        num_fonts = 100
        font_idxs = np.arange(len(data_by_domain))
        np.random.shuffle(font_idxs)
        if not denoise:
            data_by_domain = data_by_domain[font_idxs[:num_fonts]]

            print(f"NUM FONTS: {num_fonts}")
            print(f"NUM CHARS: {data_by_domain.shape[1]}")

            num_classes = data_by_domain.shape[1]
            self.all_targets = np.concatenate(
                [np.arange(num_classes)]*num_fonts, axis=0)
            self.all_domain_labels = np.repeat(np.arange(num_fonts), num_classes)
            self.all_data = data_by_domain.reshape(data_by_domain.shape[0]*data_by_domain.shape[1], data_by_domain.shape[2], data_by_domain.shape[3])

            idxs = np.arange(len(self.all_data))
            np.random.shuffle(idxs)
            train_val_max = 2600
            if num_examples > train_val_max:
                # to be able to heuristically test what happens if we have more training data
                train_val_max = 5000
            if split == 'train':
                idxs = idxs[:num_examples]
            elif split == 'val':
                idxs = idxs[num_examples: train_val_max]
            else:
                idxs = idxs[train_val_max:]
            self.targets = self.all_targets[idxs]
            self.domain_labels = self.all_domain_labels[idxs]
            self.data = self.all_data[idxs]
        else:
            # get the train data
            train_dbd = data_by_domain[font_idxs[:num_fonts]]
            all_data = train_dbd.reshape(train_dbd.shape[0]*train_dbd.shape[1], train_dbd.shape[2], train_dbd.shape[3])
            idxs = np.arange(len(all_data))
            np.random.shuffle(idxs)
            idxs = idxs[:num_examples]
            train_data = all_data[idxs]

            if num_fonts_pi is not None:
                data_by_domain = data_by_domain[font_idxs[num_fonts:num_fonts+num_fonts_pi]]
            else:
                data_by_domain = data_by_domain[font_idxs[num_fonts:]]
            self.data = data_by_domain.reshape(data_by_domain.shape[0]*data_by_domain.shape[1], data_by_domain.shape[2], data_by_domain.shape[3])
            self.data = np.concatenate([train_data, self.data], axis=0)

    def get_nearest_neighbor(self, all_imgs, x):
        idx = np.argmin(np.sum(np.square(all_imgs - x), axis=(1,2)))
        return self[idx]

    def download(self):
        if not self.download_path.exists():
            download_file_from_google_drive(self.url_id, str(self.download_path))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.denoise:
            img = self.data[index]
            img = Image.fromarray(img)
            if self.transform is not None:
                tgt_img = self.transform(img)
            if self.denoise_transform is not None:
                src_img = self.denoise_transform(img)
            return src_img, tgt_img
        else:
            img, target = self.data[index], self.targets[index]
            domain_label = self.domain_labels[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target, domain_label

    def get_item_from_all(self, index):
        img, target = self.all_data[index], self.all_targets[index]
        domain_label = self.all_domain_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, domain_label

    def __len__(self):
        return len(self.data)
