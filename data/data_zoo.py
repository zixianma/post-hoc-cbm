from torchvision import datasets
import torch
import os
import json 
import pickle
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, ToTensor, Resize, CenterCrop

def _convert_to_rgb(image):
    return image.convert('RGB')

def image_transform(
        image_size: int,
        is_train: bool,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
):
    normalize = Normalize(mean=mean, std=std)
    if is_train:
        return Compose([
            RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
    else:
        return Compose([
            Resize(image_size, interpolation=Image.BICUBIC),
            CenterCrop(image_size),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])

class ToxicCommentsDataset(Dataset):
    """Comment toxicity dataset."""

    def __init__(self, file_path, root_dir='/Users/zixianma/Desktop/Research/post-hoc-cbm/assets'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        file_path = os.path.join(self.root_dir, file_path)
        self.df = pd.read_csv(file_path) #.iloc[:640, :]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        text = self.df['comment'].iloc[idx]
        label = round(self.df['toxic_score_mean'].iloc[idx])

        sample = {'text': text, 'label': label}
        return sample
    
class HatefulMemesDataset(Dataset):
    """Hateful memes dataset."""

    def __init__(self, file_path, root_dir='/Users/zixianma/Desktop/Research/obj_functions_dev/nb_zixian', \
                 is_train=False, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform if transform else image_transform(224, is_train) 
        file_path = os.path.join(self.root_dir, file_path)
        if file_path.find('pkl') > -1:
            with open(file_path, "rb") as f:
                self.df = pickle.load(f)
        elif file_path.find('json') > -1:
            self.df = self.load_data_from_jsonl(file_path)
        else:
            raise ValueError
        
        
    def load_data_from_jsonl(self, data_path):
        f = open(data_path, )
        data_list = []
        for line in f.readlines():
            example = json.loads(line.replace('\n', ''))
            data_list.append(example)
        df = pd.DataFrame(data_list)
        return df 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.df['img'].iloc[idx])
        image = Image.open(img_name)
        text = self.df['text'].iloc[idx]
        label = self.df['label'].iloc[idx]

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'text': text, 'label': label}
        return sample
    
def get_dataset(args, preprocess=None):
    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root=args.out_dir, train=True,
                                    download=True, transform=preprocess)
        testset = datasets.CIFAR10(root=args.out_dir, train=False,
                                    download=True, transform=preprocess)
        classes = trainset.classes
        class_to_idx = {c: i for (i,c) in enumerate(classes)}
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.num_workers)
    
    
    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(root=args.out_dir, train=True,
                                    download=True, transform=preprocess)
        testset = datasets.CIFAR100(root=args.out_dir, train=False,
                                    download=True, transform=preprocess)
        classes = trainset.classes
        class_to_idx = {c: i for (i,c) in enumerate(classes)}
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.num_workers)


    elif args.dataset == "cub":
        from .cub import load_cub_data
        from .constants import CUB_PROCESSED_DIR, CUB_DATA_DIR
        from torchvision import transforms
        num_classes = 200
        TRAIN_PKL = os.path.join(CUB_PROCESSED_DIR, "train.pkl")
        TEST_PKL = os.path.join(CUB_PROCESSED_DIR, "test.pkl")
        normalizer = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        train_loader = load_cub_data([TRAIN_PKL], use_attr=False, no_img=False, 
            batch_size=args.batch_size, uncertain_label=False, image_dir=CUB_DATA_DIR, resol=224, normalizer=normalizer,
            n_classes=num_classes, resampling=True)

        test_loader = load_cub_data([TEST_PKL], use_attr=False, no_img=False, 
                batch_size=args.batch_size, uncertain_label=False, image_dir=CUB_DATA_DIR, resol=224, normalizer=normalizer,
                n_classes=num_classes, resampling=True)

        classes = open(os.path.join(CUB_DATA_DIR, "classes.txt")).readlines()
        classes = [a.split(".")[1].strip() for a in classes]
        idx_to_class = {i: classes[i] for i in range(num_classes)}
        classes = [classes[i] for i in range(num_classes)]
        print(len(classes), "num classes for cub")
        print(len(train_loader.dataset), "training set size")
        print(len(test_loader.dataset), "test set size")
        

    elif args.dataset == "ham10000":
        from .derma_data import load_ham_data
        train_loader, test_loader, idx_to_class = load_ham_data(args, preprocess)
        class_to_idx = {v:k for k,v in idx_to_class.items()}
        classes = list(class_to_idx.keys())

    elif args.dataset == "hateful_memes":
        idx_to_class = {0: 'benign', 1: 'hateful'}
        classes = ['benign', 'hateful']
        trainset = HatefulMemesDataset(file_path='hateful_memes_orig/train.jsonl') #'hm_train.pkl'
        testset = HatefulMemesDataset(file_path='hateful_memes_orig/test_unseen.jsonl') #'hm_test.pkl'
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    elif args.dataset == "toxic_comments":
        idx_to_class = {0: 'level0', 1: 'level1', 2: 'level2', 3: 'level3', 4: 'level4'}
        classes = ['level0', 'level1', 'level2', 'level3', 'level4']
        trainset = ToxicCommentsDataset(file_path='toxic_comments/train_df_avg.csv')
        testset = ToxicCommentsDataset(file_path='toxic_comments/test_df_avg.csv') 
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        raise ValueError(args.dataset)

    return train_loader, test_loader, idx_to_class, classes

