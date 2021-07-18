import warnings
warnings.filterwarnings("ignore")
import glob
import torch
import pickle
import librosa
import numpy as np

from util import *

actor_ids = np.array([str(i) for i in range(1, 25)])
actor_T = ["OAF", "YAF"]
statements = ["kids", "dogs"]
types = ["normal_intensity_1", "normal_intensity_2", "strong_intensity_1", "strong_intensity_2"]

trans_emo = {}
trans_emo["neutral"] = ["neutral", "calm"]
trans_emo["fear"] = ["fearful"]
trans_emo["happy"] = ["happy"]
trans_emo["sad"] = ["sad"]
trans_emo["angry"] = ["angry"]
trans_emo["disgust"] = ["disgust"]
trans_emo["surprised"] = ["surprised"]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_TESS, root_RAVDESS, transform, emotions, target_length, mode="train", dataset="both", classes=tuple(range(4))):
        
        self.transform = transform
        self.target_length = target_length
        self.images = []
        self.labels = []

        images = {} 
        labels = []
        
        for emotion in emotions:
            images[emotion] = []
            
            #####################
            ####### TESS ########
            #####################
            if dataset=="both" or dataset=="TESS":
                for actor in actor_T:
                    list = glob.glob(root_TESS + f"{actor}/{emotion}/*wav")
                    list.sort()
                    if mode=="train":
                        list = list[:160]
                    elif mode=="val":
                        list = list[160:180]
                    elif mode=="test":
                        list = list[180:]
                    elif mode=="all":
                        list = list
                    images[emotion] += list
                
            #####################
            ###### RAVDESS ######
            #####################
            if dataset=="both" or dataset=="RAVDESS":
                for emo_R in trans_emo[emotion]:
                    if mode=="train":
                        actors = actor_ids[:20]
                    elif mode=="val":
                        actors = actor_ids[20:22]
                    elif mode=="test":
                        actors = actor_ids[22:]
                    elif mode=="all":
                        actors = actor_ids
                    for actor in actors:
                        list = glob.glob(root_RAVDESS + f"audio/speech/{actor}/{emo_R}/*wav")
                        list.sort()
                        images[emotion] += list
            num = emotions.index(emotion)
            labels.append([num] * len(images[emotion]))
            
        # combine them together
        for label in classes:
            for image, label in zip(images[emotions[label]], labels[label]):
                self.images.append(image)
                self.labels.append(label)

    def __getitem__(self, index):
        
        image = self.images[index]
        label = self.labels[index]
        
        x, fs = librosa.load(image, None)
        x = self.transform(np.reshape(x, (x.shape[0], 1)), self.target_length)
        x = np.reshape(x, (1, x.shape[0]))
        x = torch.Tensor(x)
            
        return x, label
    
    def __len__(self):
        return len(self.images)
    
class Dataset_mel(torch.utils.data.Dataset):
    def __init__(self, root_TESS, root_RAVDESS, transform, emotions, target_length, mode="train", dataset="both", classes=tuple(range(4)), additional_transformer=True):
        
        self.transform = transform
        self.target_length = target_length
        self.additional_transformer = additional_transformer
        self.images = []
        self.labels = []

        images = {} 
        labels = []
        
        for emotion in emotions:
            images[emotion] = []
            
            #####################
            ####### TESS ########
            #####################
            if dataset=="both" or dataset=="TESS":
                for actor in actor_T:
                    list = glob.glob(root_TESS + f"feature/{actor}/{emotion}/*pkl")
                    list.sort()
                    if mode=="train":
                        list = list[:160]
                    elif mode=="val":
                        list = list[160:180]
                    elif mode=="test":
                        list = list[180:]
                    elif mode=="all":
                        list = list
                    images[emotion] += list
                
            #####################
            ###### RAVDESS ######
            #####################
            if dataset=="both" or dataset=="RAVDESS":
                for emo_R in trans_emo[emotion]:
                    if mode=="train":
                        actors = actor_ids[:20]
                    elif mode=="val":
                        actors = actor_ids[20:22]
                    elif mode=="test":
                        actors = actor_ids[22:]
                    elif mode=="all":
                        actors = actor_ids
                    for actor in actors:
                        list = glob.glob(root_RAVDESS + f"feature/speech/{actor}/{emo_R}/*pkl")
                        list.sort()
                        images[emotion] += list
            num = emotions.index(emotion)
            labels.append([num] * len(images[emotion]))
            
        # combine them together
        for label in classes:
            for image, label in zip(images[emotions[label]], labels[label]):
                self.images.append(image)
                self.labels.append(label)

    def __getitem__(self, index):
        
        image = self.images[index]
        label = self.labels[index]
        
        with open(image, mode='rb') as f:
            image = pickle.load(f).numpy()
            
        if self.transform is not None:
            image = self.transform(image)
            
        if self.additional_transformer:
            image = torch.Tensor(np.reshape(min_max(image, mean0=False), (1, image.shape[0], -1)))
            
        return image, label
    
    def __len__(self):
        return len(self.images)
