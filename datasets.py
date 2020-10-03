import torch.utils.data as data
import torch
from PIL import Image
import numpy as np
import torchvision as tv
from torchvision.datasets import MNIST, SVHN, ImageFolder, DatasetFolder
from torchvision.datasets import DatasetFolder
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity
from functools import partial
import PIL

import os
import os.path
import sys
import logging
import pydicom

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class CustomTensorDataset(data.TensorDataset):
    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors) + (index,)


class MNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        mnist_dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        # if self.train:
        #     data = mnist_dataobj.train_data
        #     target = mnist_dataobj.train_labels
        # else:
        #     data = mnist_dataobj.test_data
        #     target = mnist_dataobj.test_labels

        data = mnist_dataobj.data
        target = mnist_dataobj.targets

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        # print("mnist img:", img)
        # print("mnist target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)


class SVHN_custom(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if self.train is True:
            svhn_dataobj1 = SVHN(self.root, 'train', self.transform, self.target_transform, self.download)
            svhn_dataobj2 = SVHN(self.root, 'extra', self.transform, self.target_transform, self.download)
            data = np.concatenate((svhn_dataobj1.data, svhn_dataobj2.data), axis=0)
            target = np.concatenate((svhn_dataobj1.labels, svhn_dataobj2.labels), axis=0)

            # svhn_dataobj = SVHN(self.root, 'train', self.transform, self.target_transform, self.download)
            # data = svhn_dataobj.data
            # target = svhn_dataobj.labels
        else:
            svhn_dataobj = SVHN(self.root, 'test', self.transform, self.target_transform, self.download)
            data = svhn_dataobj.data
            target = svhn_dataobj.labels

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        # print("svhn data:", data)
        # print("len svhn data:", len(data))
        # print("type svhn data:", type(data))
        # print("svhn target:", target)
        # print("type svhn target", type(target))
        return data, target

    # def truncate_channel(self, index):
    #     for i in range(index.shape[0]):
    #         gs_index = index[i]
    #         self.data[gs_index, :, :, 1] = 0.0
    #         self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        # print("svhn img:", img)
        # print("svhn target:", target)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)


# torchvision CelebA
class CelebA_custom(VisionDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                ``identity`` (int): label for each person (data points with the same identity are the same person)
                ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                    righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)
            Defaults to ``attr``. If empty, ``None`` will be returned as target.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "celeba"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                         MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(self, root, dataidxs=None, split="train", target_type="attr", transform=None,
                 target_transform=None, download=False):
        import pandas
        super(CelebA_custom, self).__init__(root, transform=transform,
                                     target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        # if download:
        #     self.download()
        #
        # if not self._check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You can use download=True to download it')

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split = split_map[split.lower()]

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        identity = pandas.read_csv(fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0)
        bbox = pandas.read_csv(fn("list_bbox_celeba.txt"), delim_whitespace=True, header=1, index_col=0)
        landmarks_align = pandas.read_csv(fn("list_landmarks_align_celeba.txt"), delim_whitespace=True, header=1)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)

        mask = slice(None) if split is None else (splits[1] == split)

        self.filename = splits[mask].index.values
        self.identity = torch.as_tensor(identity[mask].values)
        self.bbox = torch.as_tensor(bbox[mask].values)
        self.landmarks_align = torch.as_tensor(landmarks_align[mask].values)
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)
        self.gender_index = self.attr_names.index('Male')
        self.dataidxs = dataidxs
        if self.dataidxs is None:
            self.target = self.attr[:, self.gender_index:self.gender_index + 1].reshape(-1)
        else:
            self.target = self.attr[self.dataidxs, self.gender_index:self.gender_index + 1].reshape(-1)

    def _check_integrity(self):
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))

    def download(self):
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)

        with zipfile.ZipFile(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r") as f:
            f.extractall(os.path.join(self.root, self.base_folder))

    def __getitem__(self, index):
        if self.dataidxs is None:
            X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

            target = []
            for t in self.target_type:
                if t == "attr":
                    target.append(self.attr[index, self.gender_index])
                elif t == "identity":
                    target.append(self.identity[index, 0])
                elif t == "bbox":
                    target.append(self.bbox[index, :])
                elif t == "landmarks":
                    target.append(self.landmarks_align[index, :])
                else:
                    # TODO: refactor with utils.verify_str_arg
                    raise ValueError("Target type \"{}\" is not recognized.".format(t))
        else:
            X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[self.dataidxs[index]]))

            target = []
            for t in self.target_type:
                if t == "attr":
                    target.append(self.attr[self.dataidxs[index], self.gender_index])
                elif t == "identity":
                    target.append(self.identity[self.dataidxs[index], 0])
                elif t == "bbox":
                    target.append(self.bbox[self.dataidxs[index], :])
                elif t == "landmarks":
                    target.append(self.landmarks_align[self.dataidxs[index], :])
                else:
                    # TODO: refactor with utils.verify_str_arg
                    raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)
        #print("target[0]:", target[0])
        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None
        #print("celeba target:", target)
        return X, target, index

    def __len__(self):
        if self.dataidxs is None:
            return len(self.attr)
        else:
            return len(self.dataidxs)

    def extra_repr(self):
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


class PneumoniaDataset(data.Dataset):
    """
        Pneumonia dataset that contains radiograph lung images as .dcm.
        Each patient has one image named patientId.dcm.
    """

    def __init__(self, root, subset, pIds, rescale_factor=1, dataidxs=None, transform=None, rotation_angle=0,
                 warping=False):
        """
        :param root: it has to be a path to the folder that contains the dataset folders
        :param subset: 'train' or 'test'
        :param pIds: list of patient IDs
        :param predict: boolean, if true returns images and target labels, otherwise returns only images
        :param boxes: a {patientId : list of boxes} dictionary (ex: {'pId': [[x1, y1, w1, h1], [x2, y2, w2, h2]]}
        :param rescale_factor: image rescale factor in network (image shape is supposed to be square)
        :param transform: transformation applied to the images and their target masks
        :param rotation_angle: float, defines range of random rotation angles for augmentation (-rotation_angle, +rotation_angle)
        :param warping: boolean, whether applying augmentation warping to image
        """

        # initialize variables
        self.root = os.path.expanduser(root)
        self.subset = subset
        if self.subset not in ['train', 'test']:
            raise RuntimeError('Invalid subset ' + self.subset + ', it must be one of: \'train\' or \'test\'')
        self.pIds = pIds
        # self.predict = predict
        # self.boxes = boxes
        self.rescale_factor = rescale_factor
        self.transform = transform
        self.rotation_angle = rotation_angle
        self.warping = warping

        self.data_path = self.root

        self.dataidxs = dataidxs

        self.target = torch.zeros(len(pIds),dtype=torch.long)
        # self.target = None

    def __getitem__(self, index):
        # get the corresponding pId
        if self.dataidxs is not None:
            pId = self.pIds[self.dataidxs[index]]
            target = self.target[self.dataidxs[index]]
        else:
            pId = self.pIds[index]
            target = self.target[index]
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.data_path, str(pId) + '.dcm')).pixel_array

        img = Image.fromarray(img).convert('RGB')
        # apply transforms to image
        if self.transform is not None:
            img = self.transform(img)


        return img, target, index

    def __len__(self):
        if self.dataidxs is None:
            return len(self.pIds)
        else:
            return len(self.dataidxs)

class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
            self.target = np.array(imagefolder_obj.targets)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)
            self.target = np.array(imagefolder_obj.targets)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.target[index]
        target=int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)


def make_dataset(directory, extensions=None):
    instances = []
    directory = os.path.expanduser(directory)
    if extensions is None:
        raise ValueError("Extensions cannot be None")
    if extensions is not None:
        def is_valid_file(x):
            return x.lower().endswith(extensions)

    for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if is_valid_file(path):
                instances.append(path)
    return instances


class ImageFolder_public(VisionDataset):
    def __init__(self, root, dataidxs=None, loader=default_loader, extensions=None, transform=None, target_transform=None):
        super(ImageFolder_public, self).__init__(root, transform=transform, target_transform=target_transform)
        samples = make_dataset(self.root, IMG_EXTENSIONS if extensions is None else extensions)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions
        self.dataidxs = dataidxs
        if self.dataidxs is not None:
            self.samples = np.array(samples)[self.dataidxs]
        else:
            self.samples = np.array(samples)
        self.target = np.array([0 for _ in samples])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        target = self.target[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)
