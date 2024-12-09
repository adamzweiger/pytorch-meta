import os
import json
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
import gdown
import zipfile

# Adjusted class splits as requested
CLASS_SPLITS = {
    'train': (0, 640),
    'val': (640, 800),
    'test': (800, 1000)
}

class ImagenetSketch(CombinationMetaDataset):
    """
    Imagenet-Sketch dataset with MiniImagenet-style splits, downloaded from Google Drive.
    """
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):

        if sum([meta_train, meta_val, meta_test]) > 1:
            raise ValueError("Only one of meta_train/meta_val/meta_test can be True.")
        if meta_split is None:
            if meta_train:
                meta_split = 'train'
            elif meta_val:
                meta_split = 'val'
            elif meta_test:
                meta_split = 'test'
            else:
                raise ValueError("One of meta_train/meta_val/meta_test must be True or meta_split must be provided.")

        dataset = ImagenetSketchClassDataset(
            root, meta_train=(meta_split=='train'),
            meta_val=(meta_split=='val'), meta_test=(meta_split=='test'),
            meta_split=meta_split, transform=transform, class_augmentations=class_augmentations,
            download=download
        )
        super(ImagenetSketch, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)


class ImagenetSketchClassDataset(ClassDataset):
    folder = 'imagenetsketch'
    # Original full dataset files in a single zip on Google Drive
    # After downloading and extracting this zip, we should get:
    #  - imagenet_sketch_resized.hdf5
    #  - imagenet_sketch_labels.json
    full_hdf5 = 'imagenet_sketch_resized.hdf5'
    full_labels = 'imagenet_sketch_labels.json'

    # Split files format
    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'

    # Replace this with your Google Drive file ID
    # GDRIVE_FILE_ID = "YOUR_GDRIVE_FILE_ID_HERE"
    zip_filename = 'imagenet_sketch_resized.zip'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(ImagenetSketchClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.split_filename = os.path.join(self.root,
            self.filename.format(self.meta_split))
        self.split_filename_labels = os.path.join(self.root,
            self.filename_labels.format(self.meta_split))

        self._data_file = None
        self._data = None
        self._labels = None
        self.class_names = None
        self.class_dict = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('ImagenetSketch integrity check failed for split {}.'.format(self.meta_split))

        # Once integrity is confirmed, load data
        self._initialize_data()

        self._num_classes = len(self.class_names)

    def _initialize_data(self):
        # Load class names
        with open(self.split_filename_labels, 'r') as f:
            self.class_names = json.load(f)  # list of class names
        self.class_dict = {cn: list(range(len(self.data[cn]))) for cn in self.class_names}

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data_file = h5py.File(self.split_filename, 'r')
            self._data = self._data_file['datasets']
        return self._data

    def __getitem__(self, index):
        class_name = self.class_names[index % self.num_classes]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)
        return ImagenetSketchDataset(index, self.data, class_name, transform=transform, target_transform=target_transform)

    def _check_integrity(self):
        return os.path.isfile(self.split_filename) and os.path.isfile(self.split_filename_labels)

    def close(self):
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
            self._data = None

    def download(self):
        # If files are already there, return
        if self._check_integrity():
            return

        # Check if we have the full dataset files downloaded
        if not (os.path.isfile(os.path.join(self.root, self.full_hdf5)) and 
                os.path.isfile(os.path.join(self.root, self.full_labels))):
            # Download from Google Drive
            os.makedirs(self.root, exist_ok=True)
            zip_path = os.path.join(self.root, self.zip_filename)
            if not os.path.isfile(zip_path):
                print("Downloading dataset zip from Google Drive...")
                
                # url = f"https://drive.google.com/uc?id={self.GDRIVE_FILE_ID}"
                url = "https://drive.google.com/file/d/1VcuZ2dNv91Ex5pihUsbOQh3jipXNheZG/view?usp=sharing"
                gdown.download(url, zip_path, quiet=False)
            print("Extracting dataset zip...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(self.root)

        # Create split files if not present
        for split in ['train', 'val', 'test']:
            split_data_path = os.path.join(self.root, self.filename.format(split))
            split_labels_path = os.path.join(self.root, self.filename_labels.format(split))
            if not (os.path.isfile(split_data_path) and os.path.isfile(split_labels_path)):
                self._create_split_files()

    def _create_split_files(self):
        # Load full dataset and labels
        with h5py.File(os.path.join(self.root, self.full_hdf5), 'r') as f:
            full_data = f['datasets']
            with open(os.path.join(self.root, self.full_labels), 'r') as lf:
                full_labels = json.load(lf)

        unique_classes = sorted(set(full_labels))

        # Updated class splits
        train_range = CLASS_SPLITS['train']
        val_range = CLASS_SPLITS['val']
        test_range = CLASS_SPLITS['test']

        train_classes = unique_classes[train_range[0]:train_range[1]]
        val_classes = unique_classes[val_range[0]:val_range[1]]
        test_classes = unique_classes[test_range[0]:test_range[1]]

        train_class_names = [str(c) for c in train_classes]
        val_class_names = [str(c) for c in val_classes]
        test_class_names = [str(c) for c in test_classes]

        # Group images by class
        class_to_indices = {str(c): [] for c in unique_classes}
        for i, lbl in enumerate(full_labels):
            class_to_indices[str(lbl)].append(i)

        self._write_split_files('train', train_class_names, class_to_indices, full_data)
        self._write_split_files('val', val_class_names, class_to_indices, full_data)
        self._write_split_files('test', test_class_names, class_to_indices, full_data)

    def _write_split_files(self, split, class_names, class_to_indices, full_data):
        split_data_path = os.path.join(self.root, self.filename.format(split))
        split_labels_path = os.path.join(self.root, self.filename_labels.format(split))
        with h5py.File(split_data_path, 'w') as f:
            group = f.create_group('datasets')
            for cn in class_names:
                indices = class_to_indices[cn]
                imgs = [full_data[str(i)][:] for i in indices]
                imgs = np.stack(imgs, axis=0)
                group.create_dataset(cn, data=imgs, compression="gzip")
        with open(split_labels_path, 'w') as lf:
            json.dump(class_names, lf)
        print(f"Created {split} split files.")


class ImagenetSketchDataset(Dataset):
    def __init__(self, index, data, class_name,
                 transform=None, target_transform=None):
        super(ImagenetSketchDataset, self).__init__(index, transform=transform,
                                                    target_transform=target_transform)
        self.data = data
        self.class_name = class_name
        self.images = self.data[self.class_name]

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        target = self.class_name

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (image, target)
