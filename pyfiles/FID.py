import warnings
warnings.filterwarnings("ignore")
import numpy as np
import glob
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
from scipy.linalg import sqrtm

from util import *

class inception_model():
    
    def __init__(self, model):
        layers = []
        layers += list(model.children())[:3]
        layers += [nn.MaxPool2d(3,2)]
        layers += list(model.children())[3:5]
        layers += [nn.MaxPool2d(3,2)]
        layers += list(model.children())[5:13]
        layers += list(model.children())[14:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.model = model
        
    def get(self, x, output_type="score"):
        if output_type=="feature":
            with torch.no_grad():
                x = self.feature_extractor(x)
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = F.dropout(x, training=False)
                x = torch.flatten(x, 1)
                out = x

        elif output_type =="score":
            with torch.no_grad():
                output = self.model(x)
                out = output
                
        return cuda2numpy(out)
    
def compute_FID(feat1, feat2):
    mu1, sigma1 = feat1.mean(axis=0), np.cov(feat1, rowvar=False)
    mu2, sigma2 = feat2.mean(axis=0), np.cov(feat2, rowvar=False)
    meandif = np.sum((mu1 - mu2)**2.0)
    sigma_mat, _ = sqrtm(np.dot(sigma1, sigma2), disp=False)
    fid = meandif + np.trace(sigma1 + sigma2 - 2.0*sigma_mat)
    return np.real(fid)

class FID():
    def __init__(self, batch_size=32, device=None, pretraining="ImageNet", len_classes=4):
        # pretraining = "ImageNet" or "sound"
        self.transformer = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        if device==None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.pretrainin = pretraining
        
        if pretraining=="ImageNet":
            model = torch.hub.load('pytorch/vision:v0.5.0', 'inception_v3', pretrained=True).to(self.device)
        elif pretraining=="sound":
            dir = "../notebook/instant_model_parameter/"
            model_path = f"{dir}mel_emotion_classifier_inception_lr0.001_epoch105.pth"
            model = torch.hub.load('pytorch/vision:v0.5.0', 'inception_v3', pretrained=False)
            model.fc = nn.Linear(in_features=2048, out_features=len_classes)
            model = model.to(device)
            m = torch.load(model_path, map_location=self.device) 
            model.load_state_dict(m)
            
        model.eval()
        self.im = inception_model(model)
        self.batch_size = batch_size
    
    def preprocess(self, tensor):
        images = []
        for i in range(tensor.shape[0]):
            t = tensor[i:i+1,:,:,:]
            image = image_from_output(t)[0]
            image = self.transformer(image).numpy()
            images.append(image)
        images = torch.Tensor(np.array(images))
        return images
    
    def get_all_features(self, data):
        data = self.preprocess(data)
        num = data.shape[0]
        for itr in range(int(np.ceil(num/self.batch_size))):
            input_tensor = data[itr*self.batch_size:(itr+1)*self.batch_size,:,:,:].to(self.device)
            feature = self.im.get(input_tensor, "feature")
            if itr==0:
                features = feature
            else:
                features = np.concatenate([features, feature], axis=0)
        return features
    
    def init_source_feature(self, source):
        self.feature = self.get_all_features(source)
    
    def get_FID(self, target):
        target_feature = self.get_all_features(target)
        fid = compute_FID(self.feature, target_feature)
        return fid