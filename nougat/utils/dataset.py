"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
import logging
import os
from math import prod
from pathlib import Path
from functools import partial
import random
from typing import Dict, Tuple, Callable
from PIL import Image, UnidentifiedImageError
from typing import List, Optional

import torch
import pypdf
import orjson
from torch.utils.data import Dataset
from transformers.modeling_utils import PreTrainedModel
from nougat.dataset.rasterize import rasterize_paper
import pandas as pd


class ImageDataset(torch.utils.data.Dataset):
    """
    Dataset for processing a list of images using a preparation function.

    This dataset takes a list of image paths and applies a preparation function to each image.

    Args:
        img_list (list): List of image paths.
        prepare (Callable): A preparation function to process the images.

    Attributes:
        img_list (list): List of image paths.
        prepare (Callable): The preparation function.
    """

    def __init__(self, img_list, prepare: Callable):
        super().__init__()
        self.img_list = img_list
        self.prepare = prepare

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def ignore_none_collate(batch):
        if batch is None:
            return
        try:
            batch = [x for x in batch if x is not None and x[0] is not None]
            if len(batch) == 0:
                return
            return torch.utils.data.dataloader.default_collate(batch)
        except AttributeError:
            pass

    def __getitem__(self, idx):
        try:
            img = Image.open(self.img_list[idx])
            return self.prepare(img)
        except Exception as e:
            logging.error(e)


class LazyDataset(Dataset):
    """
    Lazy loading dataset for processing PDF documents.

    This dataset allows lazy loading of PDF documents and provides access to processed images
    using a specified preparation function.

    Args:
        pdf (str): Path to the PDF document.
        prepare (Callable): A preparation function to process the images.

    Attributes:
        name (str): Name of the PDF document.
    """

    def __init__(self, pdf, prepare: Callable, pages: Optional[List[int]] = None):
        super().__init__()
        self.prepare = prepare
        self.name = str(pdf)
        self.init_fn = partial(rasterize_paper, pdf, pages=pages)
        self.dataset = None
        self.size = len(pypdf.PdfReader(pdf).pages) if pages is None else len(pages)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if i == 0 or self.dataset is None:
            self.dataset = ImageDataset(self.init_fn(), self.prepare)
        if i <= self.size and i >= 0:
            return self.dataset[i], self.name if i == self.size - 1 else ""
        else:
            raise IndexError

    @staticmethod
    def ignore_none_collate(batch):
        if batch is None:
            return None, None
        try:
            _batch = []
            for i, x in enumerate(batch):
                image, name = x
                if image is not None:
                    _batch.append(x)
                elif name:
                    if i > 0:
                        _batch[-1] = (_batch[-1][0], name)
                    elif len(batch) > 1:
                        _batch.append((batch[1][0] * 0, name))
            if len(_batch) == 0:
                return None, None
            return torch.utils.data.dataloader.default_collate(_batch)
        except AttributeError:
            pass
        return None, None


class SciPDFDataset(Dataset):
    """
    Custom dataset for scientific PDF data.

    This dataset loads data from JSONL files and provides access to images, ground truth,
    and metadata.

    Args:
        path_to_index (str): Path to the index file.
        split (str, optional): Split of the dataset (e.g., "train", "test"). Default is "train".
        root_name (str, optional): Root directory name. Default is an empty string.
        template (str, optional): Template for split naming. Default is "%s".

    Attributes:
        empty_sample: Placeholder for empty samples.
    """

    empty_sample = None

    def __init__(
        self,
        path_to_index: str,
        split: str = "train",
        root_name="",
        template="%s",
    ) -> None:
        super().__init__()
        self.path_to_index = Path(path_to_index)
        self.root_name = root_name
        self.path_to_root = self.path_to_index.parent
        if not split in self.path_to_index.stem:
            pti = self.path_to_root / (template % split + ".jsonl")
            if pti.exists():
                self.path_to_index = pti
            else:
                raise ValueError(f'Dataset file for split "{split}" not found: {pti}')
        self.dataset_file = None  # mulitprocessing
        # load seek map
        seek_path = self.path_to_root / (self.path_to_index.stem + ".seek.map")
        if seek_path.exists():
            self.seek_map = orjson.loads(seek_path.open().read())
        else:
            raise ValueError(
                'No "%s" found in %s' % (seek_path.name, str(self.path_to_root))
            )
        self.dataset_length = len(self.seek_map)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, index: int) -> Dict:
        position = self.seek_map[index]
        if self.dataset_file is None:
            self.dataset_file = self.path_to_index.open()
        self.dataset_file.seek(position)
        line = self.dataset_file.readline()
        try:
            data: Dict = orjson.loads(line)
        except Exception as e:
            logging.info(
                "JSONL for sample %i could not be loaded at position %i: %s\n%s",
                index,
                position,
                str(e),
                line,
            )
            return self.empty_sample
        img_path: Path = self.path_to_root / self.root_name / data.pop("image")
        if not img_path.exists():
            logging.info("Sample %s could not be found.", img_path)
            return self.empty_sample
        try:
            img = Image.open(img_path)
        except UnidentifiedImageError:
            logging.info("Image %s could not be opened.", img_path)
            return self.empty_sample
        return {"image": img, "ground_truth": data.pop("markdown"), "meta": data}

    def __iter__(self):
        for i in range(self.dataset_length):
            yield self[i]


class TexDataset(Dataset):
    """
    Dataset for the Image to LaTeX 100k dataset, which provides images of mathematical 
    formulas rendered on A4 pages and corresponding LaTeX representations. This dataset 
    is used to train models for image-to-LaTeX conversion tasks.

    The dataset consists of three main splits: 
    - "train": Used for training the model.
    - "validation": Used for checking the model's performance during training to monitor 
      accuracy and avoid overfitting.
    - "test": Used for final evaluation of the model's performance.

    Parameters:
    ===========
    root_path : str
        The root directory containing the dataset files.
    split : str, optional (default = "train")
        The dataset split to use. Must be one of "train", "validation", or "test".
    
    Methods:
    ========
    _get_dataframe(file_path: str) -> pd.DataFrame:
        Reads the specified dataset split file and returns a DataFrame containing the 
        formula index, image name, and render type. The formula index is replaced by the 
        corresponding LaTeX formula from the im2latex_formulas.lst file.

    __len__() -> int:
        Returns the total number of samples in the dataset.

    __getitem__(idx: int) -> Dict:
        Retrieves the image and the corresponding ground truth formula for a given index. 
        The image is loaded from the formula_images directory and converted to RGB. If the 
        image cannot be loaded, it returns None.
    """
    def __init__(self, 
                 root_path:str,
                 split: str = "train"):
        super().__init__()
        if split not in ["train", "validation", "test"]:
            raise ValueError("split should be either 'train' or 'valid' or 'test'")
        if split == "train":
            self.file_path = os.path.join(root_path, "im2latex_train.lst")
        elif split == "validation":
            self.file_path = os.path.join(root_path, "im2latex_validate.lst")
        elif split == "test":
            self.file_path = os.path.join(root_path, "im2latex_test.lst")
        
        self.img_dir = os.path.join(root_path, "formula_images")
        self.formulas_path = os.path.join(root_path, "im2latex_formulas.lst")
        self.df = self._get_dataframe(self.file_path)
    
    def _get_dataframe(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path, sep=" ", header=None, names=["formula_idx", "image_name", "render_type"])
        formulas = []
        with open(self.formulas_path, 'r', encoding='utf-8', errors='ignore') as file:
            formulas = file.readlines()
        formulas = [formula.strip() for formula in formulas]
        df['formula_idx'] = df['formula_idx'].apply(lambda idx: formulas[idx])
        return df
        
    def __len__(self) -> int:
        return len(self.df)
    def __getitem__(self, idx: int) -> Dict:
        formula = self.df['formula_idx'].iloc[idx]
        image = self.df['image_name'].iloc[idx]
        img_path = os.path.join(self.img_dir , "{}.png".format(image))
        try:
            img = Image.open(img_path).convert("RGB")
        except UnidentifiedImageError:
            logging.info("Image %s could not be opened.", img_path)
            return None
        return {"image": img, 
                "ground_truth": formula}
    
class CROHMEDataset(Dataset):
    """
    A custom PyTorch Dataset class for loading the CROHME dataset, which contains images of mathematical formulas and corresponding LaTeX labels.

    This class supports loading different splits (train, validation, test) of the CROHME dataset. The dataset includes a "caption.txt" file containing the formulas and their corresponding image IDs. Images are stored in BMP format in the associated directory. Formulas are preprocessed to remove spaces.

    Parameters
    ----------
    root_path : str
        The root directory of the dataset containing different splits and subsets.
    split : str, optional
        The dataset split to load, by default "train". The possible options are:
            - "train": Loads training data.
            - "validation": Loads validation data from a specific subset (user inputs '2014', '2016', or '2019').
            - "test": Loads test data from a specific subset (user inputs '2014', '2016', or '2019').

    Attributes
    ----------
    file_path : str
        Path to the caption file (caption.txt) containing image IDs and formula labels.
    img_dir : str
        Path to the directory containing BMP images corresponding to formulas.
    df : pd.DataFrame
        DataFrame containing image IDs and their corresponding LaTeX formulas. Spaces in formulas are removed for cleaner processing.
    
    Methods
    -------
    _get_dataframe(file_path: str) -> pd.DataFrame
        Reads the caption file and loads it into a DataFrame. Spaces in formulas are removed.
    
    __len__() -> int
        Returns the number of samples (rows) in the dataset.
    
    __getitem__(idx: int) -> Dict
        Retrieves the image and corresponding LaTeX formula for the given index.
        Returns a dictionary containing:
            - "image": The PIL image object of the BMP image.
            - "ground_truth": The corresponding LaTeX formula as a string.

    Raises
    ------
    ValueError
        If an invalid split is provided (i.e., anything other than "train", "validation", or "test").
    
    Notes
    -----
    - The `validation` and `test` splits require additional user input to select a specific subset ('2014', '2016', or '2019').
    - Images are opened using the PIL library and converted to RGB format. If the image cannot be opened, None is returned.
    """
    
    def __init__(self, 
                 root_path:str,
                 split: str = "train"):
        super().__init__()
        if split not in ["train", "validation", "test"]:
            raise ValueError("split should be either 'train' or 'valid' or 'test'")
        if split == "train":
            self.file_path = os.path.join(root_path, split, "caption.txt")
            self.img_dir = os.path.join(root_path, split, "img")
        elif split in ["validation", "test"]:
            subset = input("Enter the subset name ['2014', '2016' , '2019']: ")
            assert subset in ["2014", "2016", "2019"], "Invalid subset name. Choose from ['2014', '2016', '2019']"
            self.file_path = os.path.join(root_path, subset, "caption.txt")
            self.img_dir = os.path.join(root_path, subset, "img")
        
        self.df = self._get_dataframe(self.file_path)

    def _get_dataframe(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path, sep="\t", header=None, names=["image", "formula"])
        df["formula"] = df["formula"].str.replace(" ", "")
        return df    
    
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        formula = self.df['formula'].iloc[idx]
        image = self.df['image'].iloc[idx]
        img_path = os.path.join(self.img_dir , "{}.bmp".format(image))
        try:
            img = Image.open(img_path).convert("RGB")
        except UnidentifiedImageError:
            logging.info("Image %s could not be opened.", img_path)
            return None
        return {"image": img, 
                "ground_truth": formula}



class NougatDataset(Dataset):
    """
    Args:
        dataset_path: the path to the jsonl file
    """

    def __init__(
        self,
        dataset_path: str,
        nougat_model: PreTrainedModel,
        max_length: int,
        split: str = "train",
        root_name: str = "arxiv",
    ):
        super().__init__()
        self.nougat_model = nougat_model
        self.max_length = max_length
        self.split = split
        self.perturb = "NOUGAT_PERTURB" in os.environ and os.environ["NOUGAT_PERTURB"]
        # TODO improve naming conventions
        template = "%s"
        # self.dataset = SciPDFDataset(               # Scientific PDF dataset
        #     dataset_path, split=self.split, template=template, root_name=root_name
        # )
        self.dataset = TexDataset(dataset_path, split=self.split) # Image to Latex 100k dataset

        # self.dataset = CROHMEDataset(dataset_path, split=self.split) # CROHME dataset
        self.dataset_length = len(self.dataset)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels.
        Convert gt data into input_ids (tokenized string)

        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
        """
        sample = self.dataset[idx]
        if sample is None:
            # if sample is broken choose another randomly
            return self[random.randint(0, self.dataset_length - 1)]
        if sample is None or sample["image"] is None or prod(sample["image"].size) == 0:
            input_tensor = None
        else:
            input_tensor = self.nougat_model.encoder.prepare_input(
                sample["image"], random_padding=self.split == "train"
            )

        tokenizer_out = self.nougat_model.decoder.tokenizer(
            sample["ground_truth"],
            max_length=self.max_length,
            padding="max_length",
            return_token_type_ids=False,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenizer_out["input_ids"].squeeze(0)
        attention_mask = tokenizer_out["attention_mask"].squeeze(0)
        # randomly perturb ground truth tokens
        if self.split == "train" and self.perturb:
            # check if we perturb tokens
            unpadded_length = attention_mask.sum()
            while random.random() < 0.1:
                try:
                    pos = random.randint(1, unpadded_length - 2)
                    token = random.randint(
                        23, len(self.nougat_model.decoder.tokenizer) - 1
                    )
                    input_ids[pos] = token
                except ValueError:
                    break
        return input_tensor, input_ids, attention_mask
