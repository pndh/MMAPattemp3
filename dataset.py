import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import read_tiff
import numpy as np
import torchvision
import torchvision.transforms as transforms
import scanpy as sc
from utils import get_data
import os
import glob
from PIL import Image
import pandas as pd 
import scprep as scp
from PIL import ImageFile
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import albumentations as A
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import random
from sklearn.metrics.pairwise import cosine_similarity

class HER2ST(torch.utils.data.Dataset):
    """Some Information about HER2ST"""
    def __init__(self,train=True,gene_list=None,ds=None,fold=0):
        super(HER2ST, self).__init__()
        self.cnt_dir = 'data/her2st/data/ST-cnts'
        self.img_dir = 'data/her2st/data/ST-imgs'
        self.pos_dir = 'data/her2st/data/ST-spotfiles'
        self.lbl_dir = 'data/her2st/data/ST-pat/lbl'
        self.r = 224//2
        # gene_list = list(np.load('data/her_hvg.npy',allow_pickle=True))
        gene_list = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
        self.gene_list = gene_list
        self.names = os.listdir(self.cnt_dir)
        self.names.sort()
        self.names = [i[:2] for i in self.names]
        self.train = train
  
        # samples = ['A1','B1','C1','D1','E1','F1','G2','H1']
        samples = self.names[1:33]
        te_names = [samples[fold]]
        tr_names = list(set(samples)-set(te_names))
        if train:
            # names = names[1:33]
            # names = names[1:33] if self.cls==False else ['A1','B1','C1','D1','E1','F1','G2']
            self.names = tr_names
        else:
            # names = [names[33]]
            # names = ['A1']
            # names = [ds] if ds else ['H1']
            self.names = te_names
        print('Loading imgs...')
        self.img_dict = {i:self.get_img(i) for i in self.names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in self.names}
        

        self.label={i:None for i in self.names}

        self.lbl2id={
            'invasive cancer':0, 'breast glands':1, 'immune infiltrate':2, 
            'cancer in situ':3, 'connective tissue':4, 'adipose tissue':5, 'undetermined':-1
        }
        if not train and self.names[0] in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
            self.lbl_dict={i:self.get_lbl(i) for i in self.names}
            # self.label={i:m['label'].values for i,m in self.lbl_dict.items()}
            # print(self.lbl_dict)
            # print()
            idx=self.meta_dict[self.names[0]].index
            # print(idx)
            # print()
            lbl=self.lbl_dict[self.names[0]]
            # print(lbl)
            # print()

            lbl=lbl.loc[idx,:]['label'].values
            # print(lbl)
            # print()
            # lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
            self.label[self.names[0]]=lbl
        elif train:
            for i in self.names:
                idx=self.meta_dict[i].index
                if i in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
                    lbl=self.get_lbl(i)
                    lbl=lbl.loc[idx,:]['label'].values
                    lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
                    self.label[i]=lbl
                else:
                    self.label[i]=torch.full((len(idx),),-1)
        # self.gene_set = self.get_overlap(self.meta_dict,gene_list)
        # print(len(self.gene_set))
        # np.save('data/her_hvg',self.gene_set)
        # quit()
        
        self.gene_set = list(gene_list)
        self.exp_dict = {i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i,m in self.meta_dict.items()}
        self.center_dict = {i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) for i,m in self.meta_dict.items()}
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}


        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5,0.5,0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])
    def __getitem__(self, index):
        i = 0
        while index>=self.cumlen[i]:
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i-1]
        
        exp = self.exp_dict[self.id2name[i]][idx]
        center = self.center_dict[self.id2name[i]][idx]
        loc = self.loc_dict[self.id2name[i]][idx]

        # if self.cls or self.train==False:

        exp = torch.Tensor(exp)
        loc = torch.Tensor(loc)

        x, y = center
        patch = self.img_dict[self.id2name[i]].crop((x-self.r, y-self.r, x+self.r, y+self.r))
        if self.train:
            patch = self.transforms(patch)
        else:
            patch = transforms.ToTensor()(patch)

        if self.train:
            return patch, loc, exp
        else: 
            return patch, loc, exp, torch.Tensor(center)

    def __len__(self):
        return self.cumlen[-1]

    def get_img(self,name):
        pre = self.img_dir+'/'+name[0]+'/'+name
        fig_name = os.listdir(pre)[0]
        path = pre+'/'+fig_name
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = self.cnt_dir+'/'+name+'.tsv'
        df = pd.read_csv(path,sep='\t',index_col=0)
        return df

    def get_pos(self,name):
        path = self.pos_dir+'/'+name+'_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_lbl(self,name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        df.set_index('id',inplace=True)
        
        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))
        self.max_x = 0
        self.max_y = 0
        loc = meta[['x','y']].values
        self.max_x = max(self.max_x, loc[:,0].max())
        self.max_y = max(self.max_y, loc[:,1].max())
        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)


class UNI_HER2ST(torch.utils.data.Dataset):
    """Some Information about UNI_HER2ST"""
    def __init__(self,train=True,gene_list=None,ds=None,fold=0):
        super(UNI_HER2ST, self).__init__()
        self.cnt_dir = 'data/her2st/data/ST-cnts'
        self.img_dir = 'data/her2st/data/ST-imgs'
        self.pos_dir = 'data/her2st/data/ST-spotfiles'
        self.lbl_dir = 'data/her2st/data/ST-pat/lbl'
        self.r = 224//2
        # gene_list = list(np.load('data/her_hvg.npy',allow_pickle=True))
        gene_list = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
        self.gene_list = gene_list
        self.names = os.listdir(self.cnt_dir)
        self.names.sort()
        self.names = [i[:2] for i in self.names]
        self.train = train
  
        # samples = ['A1','B1','C1','D1','E1','F1','G2','H1']
        # samples = self.names[1:33]
        # te_names = [samples[fold]]
        # tr_names = list(set(samples)-set(te_names))
        
        # samples = self.names[0:35]
        # #test set
        # test_size = 0.2
        # te_names = sorted(list(random.sample(samples, round(test_size*len(samples)))))
        # #train set
        # tr_names = sorted(list(set(samples) - set(te_names)))
        tr_names = ['A2', 'A3', 'A4', 'A5', 'A6', 'B2', 'B3', 'B4', 'B5', 'B6', 'C2', 'C3', 'C4', 'C5', 'C6', 'D2', 'D3', 'D4', 'D5', 'D6', 'E2', 'E3', 'F2', 'F3', 'G1', 'G3', 'H2', 'H3']
        te_names = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']
        print(tr_names, te_names, sep = "\n\n")
        
        if train:
            # names = names[1:33]
            # names = names[1:33] if self.cls==False else ['A1','B1','C1','D1','E1','F1','G2']
            self.names = tr_names
        else:
            # names = [names[33]]
            # names = ['A1']
            # names = [ds] if ds else ['H1']
            self.names = te_names
        print('Loading imgs...')
        self.img_dict = {i:self.get_img(i) for i in self.names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in self.names}
        

        self.label={i:None for i in self.names}

        self.lbl2id={
            'invasive cancer':0, 'breast glands':1, 'immune infiltrate':2, 
            'cancer in situ':3, 'connective tissue':4, 'adipose tissue':5, 'undetermined':-1
        }
        if not train and self.names[0] in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
            self.lbl_dict={i:self.get_lbl(i) for i in self.names}
            # self.label={i:m['label'].values for i,m in self.lbl_dict.items()}
            # print(self.lbl_dict)
            # print()
            idx=self.meta_dict[self.names[0]].index
            # print(idx)
            # print()
            lbl=self.lbl_dict[self.names[0]]
            # print(lbl)
            # print()

            lbl=lbl.loc[idx,:]['label'].values
            # print(lbl)
            # print()
            # lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
            self.label[self.names[0]]=lbl
        elif train:
            for i in self.names:
                idx=self.meta_dict[i].index
                if i in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
                    lbl=self.get_lbl(i)
                    lbl=lbl.loc[idx,:]['label'].values
                    lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
                    self.label[i]=lbl
                else:
                    self.label[i]=torch.full((len(idx),),-1)
        # self.gene_set = self.get_overlap(self.meta_dict,gene_list)
        # print(len(self.gene_set))
        # np.save('data/her_hvg',self.gene_set)
        # quit()
        
        self.gene_set = list(gene_list)
        self.exp_dict = {i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i,m in self.meta_dict.items()}
        self.center_dict = {i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) for i,m in self.meta_dict.items()}
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}


        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        self.augmentation = A.Compose([
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.RandomBrightnessContrast(p=0.1),
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05),
        ])
    
    def center_crop(self, image, h, w, scales):
        images = [image]  
        for hat in scales[1:]:
            scale = 2 ** hat
            new_h, new_w = int(h // scale), int(w // scale)

            cropper = A.RandomCrop(width=new_w, height=new_h)
            cropped = cropper(image=image)['image']

            # Resize về kích thước ban đầu (h, w)
            resized = cv2.resize(cropped, (w, h))
            images.append(resized)

        return images
        
    def scale_crop(self, image, scales=[0, 1, 2]):
        h, w = image.shape[:2]
        images = self.center_crop(image=image, h=h, w=w, scales=scales)
            
        return images
        
    def __getitem__(self, index):
        i = 0
        while index>=self.cumlen[i]:
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i-1]
        
        exp = self.exp_dict[self.id2name[i]][idx]
        center = self.center_dict[self.id2name[i]][idx]
        loc = self.loc_dict[self.id2name[i]][idx]

        # if self.cls or self.train==False:

        exp = torch.Tensor(exp)
        loc = torch.Tensor(loc)

        x, y = center
        patch = self.img_dict[self.id2name[i]].crop((x-self.r, y-self.r, x+self.r, y+self.r))
        patch = np.array(patch)
        
        if self.train:
            patch = self.augmentation(image=patch)['image']
        
        ms_patches = self.scale_crop(patch)
        
        for i, p in enumerate(ms_patches):
            ms_patches[i] = self.transform(p).float()

        patch_0, patch_1, patch_2 = ms_patches
        
        if self.train:
            return patch_0, patch_1, patch_2, loc, exp
        else: 
            return patch_0, patch_1, patch_2, loc, exp, torch.Tensor(center)

    def __len__(self):
        return self.cumlen[-1]

    def get_img(self,name):
        pre = self.img_dir+'/'+name[0]+'/'+name
        fig_name = os.listdir(pre)[0]
        path = pre+'/'+fig_name
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = self.cnt_dir+'/'+name+'.tsv'
        df = pd.read_csv(path,sep='\t',index_col=0)
        return df

    def get_pos(self,name):
        path = self.pos_dir+'/'+name+'_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_lbl(self,name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        df.set_index('id',inplace=True)
        
        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))
        self.max_x = 0
        self.max_y = 0
        loc = meta[['x','y']].values
        self.max_x = max(self.max_x, loc[:,0].max())
        self.max_y = max(self.max_y, loc[:,1].max())
        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)
    

class WSUNI_HER2ST(torch.utils.data.Dataset):
    """Some Information about UNI_HER2ST"""
    def __init__(self,train=True,gene_list=None,ds=None,fold=0, cache_dir='cache_features/', topk=16):
        super(WSUNI_HER2ST, self).__init__()
        self.cnt_dir = 'data/her2st/data/ST-cnts'
        self.img_dir = 'data/her2st/data/ST-imgs'
        self.pos_dir = 'data/her2st/data/ST-spotfiles'
        self.lbl_dir = 'data/her2st/data/ST-pat/lbl'
        self.r = 224//2
        gene_list = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
        self.gene_list = gene_list
        self.names = os.listdir(self.cnt_dir)
        self.names.sort()
        self.names = [i[:2] for i in self.names]
        self.train = train
  
        tr_names = ['A2', 'A3', 'A4', 'A5', 'A6', 'B2', 'B3', 'B4', 'B5', 'B6', 'C2', 'C3', 'C4', 'C5', 'C6', 'D2', 'D3', 'D4', 'D5', 'D6', 'E2', 'E3', 'F2', 'F3', 'G1', 'G3', 'H2', 'H3']
        te_names = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']
        print(tr_names, te_names, sep = "\n\n")
        
        if train:
            self.names = tr_names
        else:
            self.names = te_names
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in self.names}
        

        self.label={i:None for i in self.names}

        self.lbl2id={
            'invasive cancer':0, 'breast glands':1, 'immune infiltrate':2, 
            'cancer in situ':3, 'connective tissue':4, 'adipose tissue':5, 'undetermined':-1
        }
        if not train:
            self.lbl_dict={i:self.get_lbl(i) for i in self.names} 
            # self.label={i:m['label'].values for i,m in self.lbl_dict.items()}

            # print(self.lbl_dict)
            # print()
            for i in self.names:
                idx=self.meta_dict[i].index
            # print(idx)
            # print()
                lbl=self.lbl_dict[i]
            # print(lbl)
            # print()
                lbl=lbl.loc[idx,:]['label'].values
            # print(lbl)
            # print()
            # lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
                self.label[i]=lbl
        elif train:
            for i in self.names:
                idx=self.meta_dict[i].index
                if i in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
                    lbl=self.get_lbl(i)

                    print(lbl)
                    print("finish")

                    lbl=lbl.loc[idx,:]['label'].values
                    lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
                    self.label[i]=lbl
                else:
                    self.label[i]=torch.full((len(idx),),-1)
        
        self.gene_set = list(gene_list)
        self.exp_dict = {i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i,m in self.meta_dict.items()}
        self.center_dict = {i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) for i,m in self.meta_dict.items()}
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}


        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))
        
        self.topk = topk
        self.cache_dir = cache_dir

        self.cls_tokens = {n: np.load(f"{cache_dir}/{n}_clss.npy") for n in self.names}
        self.out_tokens = {n: np.load(f"{cache_dir}/{n}_outs.npy") for n in self.names}
        self.cluster_centers = {n: np.load(f"{cache_dir}/{n}_cluster_centers.npy") for n in self.names}
        self.cluster_centers_loc = {n: np.load(f"{cache_dir}/{n}_cluster_centers_loc_mean.npy") for n in self.names}
        
    def __getitem__(self, index):
        i = 0
        while index>=self.cumlen[i]:
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i-1]
        
        name = self.id2name[i]
        exp = self.exp_dict[name][idx]
        center = self.center_dict[name][idx]
        loc = self.loc_dict[name][idx]
        cls_token = self.cls_tokens[name][idx]
        out_token = self.out_tokens[name][idx]

        sims = cosine_similarity(cls_token[None, :], self.cluster_centers[name])[0]
        n_clusters = len(self.cluster_centers[name])
        adaptive_topk = int(0.5 * n_clusters)
        # top_idx = np.argsort(-sims)[:self.topk]
        top_idx = np.argsort(-sims)[:adaptive_topk]
        neighbor_centers = self.cluster_centers[name][top_idx]
        neighbor_centers_loc = self.cluster_centers_loc[name][top_idx]
        
        pad_size = self.topk - neighbor_centers.shape[0]
        if pad_size > 0:
            pad_tensor = torch.zeros((pad_size, 1024))
            pad_tensor_loc = torch.zeros((pad_size, 2))
            neighbor_centers = np.vstack([neighbor_centers, pad_tensor.numpy()])
            neighbor_centers_loc = np.vstack([neighbor_centers_loc, pad_tensor_loc.numpy()])

        if self.train:
            return torch.tensor(out_token).float(), torch.tensor(cls_token).float(), torch.tensor(neighbor_centers).float(), torch.tensor(neighbor_centers_loc).float(), torch.tensor(center).float(), torch.tensor(exp).float()
        else:
            return torch.tensor(out_token).float(), torch.tensor(cls_token).float(), torch.tensor(neighbor_centers).float(), torch.tensor(neighbor_centers_loc).float(), torch.tensor(loc).float(), torch.tensor(exp).float(), torch.tensor(center).float()

    def __len__(self):
        return self.cumlen[-1]

    def get_cnt(self,name):
        path = self.cnt_dir+'/'+name+'.tsv'
        df = pd.read_csv(path,sep='\t',index_col=0)
        return df

    def get_pos(self,name):
        path = self.pos_dir+'/'+name+'_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_lbl(self,name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        df.set_index('id',inplace=True)
        
        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))
        self.max_x = 0
        self.max_y = 0
        loc = meta[['x','y']].values
        self.max_x = max(self.max_x, loc[:,0].max())
        self.max_y = max(self.max_y, loc[:,1].max())
        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)

class WSUNI_SliceLevel_HER2ST(torch.utils.data.Dataset):
    """Some Information about UNI_HER2ST"""
    def __init__(self,train=True,gene_list=None,ds=None,fold=0, cache_dir='cache_features/', topk=16):
        super(WSUNI_SliceLevel_HER2ST, self).__init__()
        self.cnt_dir = 'data/her2st/data/ST-cnts'
        self.img_dir = 'data/her2st/data/ST-imgs'
        self.pos_dir = 'data/her2st/data/ST-spotfiles'
        self.lbl_dir = 'data/her2st/data/ST-pat/lbl'
        self.r = 224//2
        gene_list = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
        self.gene_list = gene_list
        self.names = os.listdir(self.cnt_dir)
        self.names.sort()
        self.names = [i[:2] for i in self.names]
        self.train = train
  
        tr_names = ['A2', 'A3', 'A4', 'A5', 'A6', 'B2', 'B3', 'B4', 'B5', 'B6', 'C2', 'C3', 'C4', 'C5', 'C6', 'D2', 'D3', 'D4', 'D5', 'D6', 'E2', 'E3', 'F2', 'F3', 'G1', 'G3', 'H2', 'H3']
        # te_names = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']
        te_names = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']
        print(tr_names, te_names, sep = "\n\n")
        
        if train:
            self.names = tr_names
        else:
            self.names = te_names
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in self.names}
        

        self.label={i:None for i in self.names}

        self.lbl2id={
            'invasive cancer':0, 'breast glands':1, 'immune infiltrate':2, 
            'cancer in situ':3, 'connective tissue':4, 'adipose tissue':5, 'undetermined':-1
        }
        if not train:
            self.lbl_dict={i:self.get_lbl(i) for i in self.names} 
            # self.label={i:m['label'].values for i,m in self.lbl_dict.items()}

            # print(self.lbl_dict)
            # print()
            for i in self.names:
                idx=self.meta_dict[i].index
            # print(idx)
            # print()
                lbl=self.lbl_dict[i]
            # print(lbl)
            # print()
                lbl=lbl.loc[idx,:]['label'].values
            # print(lbl)
            # print()
            # lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
                self.label[i]=lbl
        elif train:
            for i in self.names:
                idx=self.meta_dict[i].index
                if i in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
                    lbl=self.get_lbl(i)

                    print(lbl)
                    print("finish")

                    lbl=lbl.loc[idx,:]['label'].values
                    lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
                    self.label[i]=lbl
                else:
                    self.label[i]=torch.full((len(idx),),-1)
        
        self.gene_set = list(gene_list)
        self.exp_dict = {i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i,m in self.meta_dict.items()}
        self.center_dict = {i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) for i,m in self.meta_dict.items()}
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}


        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))
        
        self.topk = topk
        self.cache_dir = cache_dir

        self.cls_tokens = {n: np.load(f"{cache_dir}/{n}_clss.npy") for n in self.names}
        self.out_tokens = {n: np.load(f"{cache_dir}/{n}_outs.npy") for n in self.names}
        self.cluster_centers = {n: np.load(f"{cache_dir}/{n}_cluster_centers.npy") for n in self.names}
        self.cluster_centers_loc = {n: np.load(f"{cache_dir}/{n}_cluster_centers_loc_mean.npy") for n in self.names}
        
    def __getitem__(self, index):
        name = self.id2name[index]
        exps = self.exp_dict[name]                         # [num_patches, num_genes]
        centers = self.center_dict[name]                   # [num_patches, 2]
        locs = self.loc_dict[name]                         # [num_patches, 2]
        cls_tokens = self.cls_tokens[name]                 # [num_patches, 1024]
        out_tokens = self.out_tokens[name]                 # [num_patches, out_dim]
        cluster_centers = self.cluster_centers[name]       # [num_clusters, 1024]
        cluster_centers_loc = self.cluster_centers_loc[name]
            
        sims = cosine_similarity(cls_tokens, cluster_centers)  # [num_patches, num_clusters]
        n_clusters = cluster_centers.shape[0]
        adaptive_topk = int(0.5 * n_clusters)

        # Get topk similar cluster centers for each patch
        top_indices = np.argsort(-sims, axis=1)[:, :adaptive_topk]  # [num_patches, adaptive_topk]
        
        neighbor_centers_list = []
        neighbor_centers_loc_list = []
        
        for i in range(len(top_indices)):
            neighbors = cluster_centers[top_indices[i]]  # [adaptive_topk, 1024]
            neighbors_loc = cluster_centers_loc[top_indices[i]]
            pad_size = self.topk - neighbors.shape[0]
            if pad_size > 0:
                pad_tensor = np.zeros((pad_size, 1024))
                neighbors = np.vstack([neighbors, pad_tensor])
                pad_tensor_loc = np.zeros((pad_size, 2))
                neighbors_loc = np.vstack([neighbors_loc, pad_tensor_loc])
            neighbor_centers_list.append(neighbors)
            neighbor_centers_loc_list.append(neighbors_loc)
        
        neighbor_centers = torch.tensor(np.stack(neighbor_centers_list))  # [num_patches, topk, 1024]
        neighbor_centers_loc = torch.tensor(np.stack(neighbor_centers_loc_list))

        if self.train:
            return torch.tensor(out_tokens).float(), \
                torch.tensor(cls_tokens).float(), \
                neighbor_centers.float(), \
                neighbor_centers_loc.float(), \
                torch.tensor(centers).float(), \
                torch.tensor(exps).float()
        else:
            return torch.tensor(out_tokens).float(), \
                torch.tensor(cls_tokens).float(), \
                neighbor_centers.float(), \
                neighbor_centers_loc.float(), \
                torch.tensor(locs).float(), \
                torch.tensor(exps).float(), \
                torch.tensor(centers).float()   

    def __len__(self):
        return len(self.exp_dict)

    def get_cnt(self,name):
        path = self.cnt_dir+'/'+name+'.tsv'
        df = pd.read_csv(path,sep='\t',index_col=0)
        return df

    def get_pos(self,name):
        path = self.pos_dir+'/'+name+'_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_lbl(self,name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        df.set_index('id',inplace=True)
        
        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))
        self.max_x = 0
        self.max_y = 0
        loc = meta[['x','y']].values
        self.max_x = max(self.max_x, loc[:,0].max())
        self.max_y = max(self.max_y, loc[:,1].max())
        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)

class ViT_HER2ST(torch.utils.data.Dataset):
    def __init__(self,train=True,gene_list=None,ds=None,sr=False,fold=0):
        super(ViT_HER2ST, self).__init__()

        #data for the 36 breast cancer sections used in this study
        self.cnt_dir = 'data/her2st/data/ST-cnts' 

        #accompanying histology images
        self.img_dir = 'data/her2st/data/ST-imgs' 

        #list of selected spots for each case, used to subset the raw gene count matrices
        self.pos_dir = 'data/her2st/data/ST-spotfiles'

        #the label with corresponding coordinates of each spot. originally extracted from the annotated HE images
        self.lbl_dir = 'data/her2st/data/ST-pat/lbl' 

        self.r = 224//4 #patch size / 2, here, r = 56 --> 112x112 patches

        #785 genes, filtered from 1000 original - remove gene if they appear in less than 1000 spots
        
        gene_list = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True)) 
        self.gene_list = gene_list
        self.names = os.listdir(self.cnt_dir)
        self.names.sort()
        self.names = [i[:2] for i in self.names]
        self.train = train
        self.sr = sr
        #divide into train/test sets
        # samples = ['A1','B1','C1','D1','E1','F1','G2','H1']
        samples = self.names[1:33]
        #test set - leave one out cross validation
        te_names = [samples[fold]]

        #train set
        tr_names = list(set(samples)-set(te_names))

        if train:
            # names = names[1:33]
            # names = names[1:33] if self.cls==False else ['A1','B1','C1','D1','E1','F1','G2']
            self.names = tr_names
        else:
            # names = [names[33]]
            # names = ['A1']
            # names = [ds] if ds else ['H1']
            self.names = te_names
        
        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in self.names}

        #metadata is a table, including,for each selected spot, coordinates, the number of genes present (per gene type)
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in self.names}

        self.label={i:None for i in self.names}

        self.lbl2id={
            'invasive cancer':0, 'breast glands':1, 'immune infiltrate':2, 
            'cancer in situ':3, 'connective tissue':4, 'adipose tissue':5, 'undetermined':-1
        }
        if not train and self.names[0] in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
            self.lbl_dict={i:self.get_lbl(i) for i in self.names}
            # self.label={i:m['label'].values for i,m in self.lbl_dict.items()}

            # print(self.lbl_dict)
            # print()
            idx=self.meta_dict[self.names[0]].index
            # print(idx)
            # print()
            lbl=self.lbl_dict[self.names[0]]
            # print(lbl)
            # print()
            lbl=lbl.loc[idx,:]['label'].values
            # print(lbl)
            # print()
            # lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
            self.label[self.names[0]]=lbl
        elif train:
            for i in self.names:
                idx=self.meta_dict[i].index
                if i in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
                    lbl=self.get_lbl(i)

                    print(lbl)
                    print("finish")

                    lbl=lbl.loc[idx,:]['label'].values
                    lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
                    self.label[i]=lbl
                else:
                    self.label[i]=torch.full((len(idx),),-1)




        self.gene_set = list(gene_list)

        #gene expression data, normalized and converted to natural log scale
        self.exp_dict = {i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i,m in self.meta_dict.items()}

        #get pixel coordinates, rounded down
        self.center_dict = {i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) for i,m in self.meta_dict.items()}

        #get coordinates (physical distance, in tissue)
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}


        self.lengths = [len(i) for i in self.meta_dict.values()] #num of spots per sample
        self.cumlen = np.cumsum(self.lengths) #cumulative indexing, considered as global index per sample
        self.id2name = dict(enumerate(self.names)) #index-sample name mapping

        #image augmentation
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5,0.5,0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])

    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i,exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp>0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:,j])


    def __getitem__(self, index):
        i = index
        im = self.img_dict[self.id2name[i]]
        #switch back to (H, W, C) format
        im = im.permute(1,0,2)
        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        loc = self.loc_dict[self.id2name[i]]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4 


        # output:

        # for normal mode
        # patches: each row is a flattened image patch --> dim = [num_spots, len of row aka 3x112x112]
        # positions: physical coordinates (x,y) of each spot --> dim =  [num_spots, 2]
        # exps: gene expression values of each spot --> dim = [num_spots, total_num_genes aka 785]

        # for sr - additional steps:
        # define bounds for patch subsampling, then subsample patches by grid. here, gri d contains 30x30 sub-patches
        if self.sr:
            centers = torch.LongTensor(centers)
            max_x = centers[:,0].max().item()
            max_y = centers[:,1].max().item()
            min_x = centers[:,0].min().item()
            min_y = centers[:,1].min().item()
            r_x = (max_x - min_x)//30
            r_y = (max_y - min_y)//30

            centers = torch.LongTensor([min_x,min_y]).view(1,-1)
            positions = torch.LongTensor([0,0]).view(1,-1)
            x = min_x
            y = min_y

            while y < max_y:  
                x = min_x            
                while x < max_x:
                    centers = torch.cat((centers,torch.LongTensor([x,y]).view(1,-1)),dim=0)
                    positions = torch.cat((positions,torch.LongTensor([x//r_x,y//r_y]).view(1,-1)),dim=0)
                    x += 56                
                y += 56
            
            centers = centers[1:,:]
            positions = positions[1:,:]

            n_patches = len(centers)
            patches = torch.zeros((n_patches,patch_dim))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                patches[i] = patch.flatten()

            
            return patches, positions, centers

        else:    
            n_patches = len(centers)
            
            patches = torch.zeros((n_patches,patch_dim))
            exps = torch.Tensor(exps)


            for i in range(n_patches):
                center = centers[i]
                x, y = center
                #patch of size 2rx2rx3
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                patches[i] = patch.flatten()

           
            if self.train:
                return patches, positions, exps
            else: 
                return patches, positions, exps, torch.Tensor(centers)
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        pre = self.img_dir+'/'+name[0]+'/'+name
        fig_name = os.listdir(pre)[0]
        path = pre+'/'+fig_name
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = self.cnt_dir+'/'+name+'.tsv'
        df = pd.read_csv(path,sep='\t',index_col=0)

        return df

    def get_pos(self,name):
        path = self.pos_dir+'/'+name+'_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        # id implies coordinate: (x,y)
        return df

    def get_lbl(self,name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        df.set_index('id',inplace=True)

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))
        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)
    

class WS_UNI_HER2ST(torch.utils.data.Dataset):
    def __init__(self,train=True,gene_list=None,ds=None,sr=False,fold=0):
        super(WS_UNI_HER2ST, self).__init__()

        #data for the 36 breast cancer sections used in this study
        self.cnt_dir = 'data/her2st/data/ST-cnts' 

        #accompanying histology images
        self.img_dir = 'data/her2st/data/ST-imgs' 

        #list of selected spots for each case, used to subset the raw gene count matrices
        self.pos_dir = 'data/her2st/data/ST-spotfiles'

        #the label with corresponding coordinates of each spot. originally extracted from the annotated HE images
        self.lbl_dir = 'data/her2st/data/ST-pat/lbl' 

        self.r = 224//2 #patch size / 2, here, r = 56 --> 112x112 patches

        #785 genes, filtered from 1000 original - remove gene if they appear in less than 1000 spots
        
        gene_list = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True)) 
        self.gene_list = gene_list
        self.names = os.listdir(self.cnt_dir)
        self.names.sort()
        self.names = [i[:2] for i in self.names]
        self.train = train
        self.sr = sr
        #divide into train/test sets
        # samples = ['A1','B1','C1','D1','E1','F1','G2','H1']
        samples = self.names[1:33]
        #test set - leave one out cross validation
        te_names = [samples[fold]]

        #train set
        tr_names = list(set(samples)-set(te_names))

        if train:
            # names = names[1:33]
            # names = names[1:33] if self.cls==False else ['A1','B1','C1','D1','E1','F1','G2']
            self.names = tr_names
        else:
            # names = [names[33]]
            # names = ['A1']
            # names = [ds] if ds else ['H1']
            self.names = te_names
        
        print('Loading imgs...')
        self.img_dict = {i:self.get_img(i) for i in self.names}

        #metadata is a table, including,for each selected spot, coordinates, the number of genes present (per gene type)
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in self.names}

        self.label={i:None for i in self.names}

        self.lbl2id={
            'invasive cancer':0, 'breast glands':1, 'immune infiltrate':2, 
            'cancer in situ':3, 'connective tissue':4, 'adipose tissue':5, 'undetermined':-1
        }
        if not train and self.names[0] in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
            self.lbl_dict={i:self.get_lbl(i) for i in self.names}
            # self.label={i:m['label'].values for i,m in self.lbl_dict.items()}

            # print(self.lbl_dict)
            # print()
            idx=self.meta_dict[self.names[0]].index
            # print(idx)
            # print()
            lbl=self.lbl_dict[self.names[0]]
            # print(lbl)
            # print()
            lbl=lbl.loc[idx,:]['label'].values
            # print(lbl)
            # print()
            # lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
            self.label[self.names[0]]=lbl
        elif train:
            for i in self.names:
                idx=self.meta_dict[i].index
                if i in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
                    lbl=self.get_lbl(i)

                    print(lbl)
                    print("finish")

                    lbl=lbl.loc[idx,:]['label'].values
                    lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
                    self.label[i]=lbl
                else:
                    self.label[i]=torch.full((len(idx),),-1)

        self.gene_set = list(gene_list)

        #gene expression data, normalized and converted to natural log scale
        self.exp_dict = {i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i,m in self.meta_dict.items()}

        #get pixel coordinates, rounded down
        self.center_dict = {i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) for i,m in self.meta_dict.items()}

        #get coordinates (physical distance, in tissue)
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}


        self.lengths = [len(i) for i in self.meta_dict.values()] #num of spots per sample
        self.cumlen = np.cumsum(self.lengths) #cumulative indexing, considered as global index per sample
        self.id2name = dict(enumerate(self.names)) #index-sample name mapping

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    
    def center_crop(self, image, h, w, scales):
        images = [image]  
        for hat in scales[1:]:
            scale = 2 ** hat
            new_h, new_w = int(h // scale), int(w // scale)

            cropper = A.RandomCrop(width=new_w, height=new_h)
            cropped = cropper(image=image)['image']

            # Resize về kích thước ban đầu (h, w)
            resized = cv2.resize(cropped, (w, h))
            images.append(resized)

        return images

    def scale_crop(self, image, scales=[0, 1, 2]):
        h, w = image.shape[:2]
        images = self.center_crop(image=image, h=h, w=w, scales=scales)
            
        return images
    
    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i,exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp>0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:,j])


    def __getitem__(self, index):
        i = index
        im = self.img_dict[self.id2name[i]]
        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        loc = self.loc_dict[self.id2name[i]]
        positions = torch.LongTensor(loc)

        # output:

        # for normal mode
        # patches: each row is a flattened image patch --> dim = [num_spots, len of row aka 3x112x112]
        # positions: physical coordinates (x,y) of each spot --> dim =  [num_spots, 2]
        # exps: gene expression values of each spot --> dim = [num_spots, total_num_genes aka 785]

        # for sr - additional steps:
        # define bounds for patch subsampling, then subsample patches by grid. here, gri d contains 30x30 sub-patches
        if self.sr:
            centers = torch.LongTensor(centers)
            max_x = centers[:,0].max().item()
            max_y = centers[:,1].max().item()
            min_x = centers[:,0].min().item()
            min_y = centers[:,1].min().item()
            r_x = (max_x - min_x)//30
            r_y = (max_y - min_y)//30

            centers = torch.LongTensor([min_x,min_y]).view(1,-1)
            positions = torch.LongTensor([0,0]).view(1,-1)
            x = min_x
            y = min_y

            while y < max_y:  
                x = min_x            
                while x < max_x:
                    centers = torch.cat((centers,torch.LongTensor([x,y]).view(1,-1)),dim=0)
                    positions = torch.cat((positions,torch.LongTensor([x//r_x,y//r_y]).view(1,-1)),dim=0)
                    x += 56                
                y += 56
            
            centers = centers[1:,:]
            positions = positions[1:,:]

            patches_0, patches_1, patches_2 = [], [], []

            for center in centers:
                x, y = center
                patch = im.crop((x-self.r, y-self.r, x+self.r, y+self.r))
                patch = np.array(patch)

                ms_patches = self.scale_crop(patch)

                for i, p in enumerate(ms_patches):
                    ms_patches[i] = self.transform(p).float()
                
                patch_0, patch_1, patch_2 = ms_patches
            
                patches_0.append(patch_0)
                patches_1.append(patch_1)
                patches_2.append(patch_2)

            # Chuyển list → tensor
            patches_0 = torch.stack(patches_0)
            patches_1 = torch.stack(patches_1)
            patches_2 = torch.stack(patches_2)

            return patches_0, patches_1, patches_2, positions, centers

        else:    
            exps = torch.Tensor(exps)
            patches_0, patches_1, patches_2 = [], [], []

            for center in centers:
                x, y = center
                patch = im.crop((x-self.r, y-self.r, x+self.r, y+self.r))
                patch = np.array(patch)
    
                ms_patches = self.scale_crop(patch)
 
                for i, p in enumerate(ms_patches):
                    ms_patches[i] = self.transform(p).float()
                
                patch_0, patch_1, patch_2 = ms_patches
            
                patches_0.append(patch_0)
                patches_1.append(patch_1)
                patches_2.append(patch_2)

            # Chuyển list → tensor
            patches_0 = torch.stack(patches_0)
            patches_1 = torch.stack(patches_1)
            patches_2 = torch.stack(patches_2)
           
            if self.train:
                return patches_0, patches_1, patches_2, positions, exps
            else: 
                return patches_0, patches_1, patches_2, positions, exps, torch.Tensor(centers)
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        pre = self.img_dir+'/'+name[0]+'/'+name
        fig_name = os.listdir(pre)[0]
        path = pre+'/'+fig_name
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = self.cnt_dir+'/'+name+'.tsv'
        df = pd.read_csv(path,sep='\t',index_col=0)

        return df

    def get_pos(self,name):
        path = self.pos_dir+'/'+name+'_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        # id implies coordinate: (x,y)
        return df

    def get_lbl(self,name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        df.set_index('id',inplace=True)

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))
        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)

if __name__ == "__main__":
    dataset = HER2ST(train=True, fold=5)
    sample = dataset[0]