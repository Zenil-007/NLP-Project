import os
import json
import re
from PIL import Image
import webdataset as wds
import random
from torch.utils.data import Dataset
from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset


class MIMICDataset(Dataset):
    def __init__(self, vis_processor=None, text_processor=None, image_root=None, ann_path=None):
        self.image_root = image_root
        self.ann_path = ann_path
        
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        # load annotation file
        with open(ann_path, 'r') as f:
            self.annotations = json.load(f)
        self.train_data = self.annotations['train']
       
    def __len__(self):
        return len(self.train_data)
        
    def __getitem__(self, index):
        data_sample = self.train_data[index]
        image_path = data_sample['image_path']
        
        # load image
        image_id = data_sample['id']
        image = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
        image = self.vis_processor(image)
        
        # load caption
        caption = data_sample['report']
        caption = self.clean_reports(caption)
        
        return {"image": image,
                "text_input": caption,
                "image_id": image_id}
        
    def clean_reports(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report
        
class MIMICGenerateThenRefineDataset(Dataset):
    def __init__(self, vis_processor=None, text_processor=None, image_root=None, ann_path=None, unlabeled_ann_path=None, retrieval_size=3):
        self.image_root = image_root
        self.ann_path = ann_path
        self.retrieval_size = retrieval_size
        
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        # load annotation file
        with open(ann_path, 'r') as f:
            self.annotations = json.load(f)
        self.train_data = self.annotations['train']
       
        # load unlabeled data
        self.unlabeled_data_list = []
        with open(unlabeled_ann_path, 'r') as f:
            for line in f.readlines:
                self.unlabeled_data_list.append(line.strip('\n'))
            
        import random
        self.unlabeled_data_list = random.sample(self.unlabeled_data_list, 3000)
            
        print(f"There are total {len(self.unlabeled_data_list)} unlabeled reports.")
       
    def __len__(self):
        return len(self.train_data)
        
    def __getitem__(self, index):
        data = self.train_data[index]
        data_samples = random.sample(self.train_data, self.retrieval_size - 1)
        image_path = data['image_path']
        
        # load image
        image_id = data['id']
        image = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
        image = self.vis_processor(image)
        
        # load caption
        caption = data['report']
        caption = self.clean_reports(caption)
        
        # load reference caption
        all_ref_captions = []
        ref_caption = data['ref_report']
        ref_caption = self.clean_reports(ref_caption)
        all_ref_captions.append(ref_caption)
        
        for data_sample in data_samples:
            ref_caption = data_sample['ref_report']
            ref_caption = self.clean_reports(ref_caption)
            all_ref_captions.append(ref_caption)
        
        # load unlabeled caption
        unlabeled_caption = random.sample(self.unlabeled_data_list, self.retrieval_size)
        
        return {"image": image,
                "text_input": caption,
                "ref_caption": ref_caption,
                "unlabeled_caption": unlabeled_caption,
                "image_id": image_id}
        
    def clean_reports(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report
    
    