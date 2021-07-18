import warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
import IPython
import librosa
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn

def cuda2numpy(x):
    return x.detach().to("cpu").numpy()

def cuda2cpu(x):
    return x.detach().to("cpu")

def my_round(x, deg):
    return round(x * 10**deg)/10**deg

def relu_numpy(x):
    return np.array(F.relu(torch.tensor(x)))

def min_max(x, axis=None, mean0=False, get_param=False):
#     min = x.min(axis=axis, keepdims=True)
#     max = x.max(axis=axis, keepdims=True)
    min = x.min()
    max = x.max()
    result = (x-min)/(max-min+1e-8)
    if mean0 :
        result = result*2 - 1
    if get_param:
        return result, min, max
    return result

def min_max_torch(x, mean0=False):
    min_ = torch.min(x)
    max_ = torch.max(x)
    result = (x-min_)/(max_-min_+1e-8)
    if mean0 :
        result = result*2 - 1
    return result

def min_max_normalize(x, target_min, target_max, axis=None):
    min_ = x.min(axis=axis, keepdims=True)
    max_ = x.max(axis=axis, keepdims=True)
    result = (x-min_)/(max_-min_+1e-8)
    result = result*(target_max-target_min) + target_min
    return result

def min_max_normalize_torch(x, target_min, target_max, axis=None):
    if type(axis)==list:
        min_ = x
        for dim in axis[::-1]:
            min_ = torch.min(min_, dim=dim, keepdim=True).values
        max_ = x
        for dim in axis[::-1]:
            max_ = torch.max(max_, dim=dim, keepdim=True).values
    elif type(axis)==int:
        min_ = torch.min(x, dim=axis, keepdim=True).values
        max_ = torch.max(x, dim=axis, keepdim=True).values
    else:
        min_ = torch.min(x)
        max_ = torch.max(x)
    result = (x-min_)/(max_-min_+1e-8)
    result = result*(target_max-target_min) + target_min
    return result

def mel_normalize(data, data_type="LJSpeech"):
    if data_type=="LJSpeech":
        min, max = -11.422722816467285, 0.6957301956581059
    return min_max_normalize(data, min, max)

def mel_normalize_torch(data, data_type="LJSpeech", axis=None):
    if data_type=="LJSpeech":
        min, max = -11.422722816467285, 0.6957301956581059
    return min_max_normalize_torch(data, min, max, axis=axis)

def image_from_output(output):
    image_list = []
    output = output.detach().to("cpu").numpy()
    for i in range(output.shape[0]):
        a = output[i]
        a = np.tile(np.transpose(a, axes=(1,2,0)), (1,1,int(3/a.shape[0])))
        a = min_max(a)*2**8 
        a[a>255] = 255
        a = np.uint8(a)
        a = Image.fromarray(a)
        image_list.append(a)
    return image_list

def play_audio(data, rate):
    IPython.display.display(IPython.display.Audio(data=data,rate=rate))
    
def silence_removal(x, top_db=25, trim_window=256, trim_stride=128, only_edge=True, top_db_intermediate=50):
    nonsilence = librosa.effects.split(x, top_db=top_db, frame_length=trim_window, hop_length=trim_stride)
    x_rec = np.array([])
    x_remove_edge = x[nonsilence[0][0]:nonsilence[-1][1]]
    if only_edge:
        x_rec = x_remove_edge
    else:
        nonsilence_low = librosa.effects.split(x_remove_edge, top_db=top_db_intermediate, frame_length=trim_window, hop_length=trim_stride)
        for i in range(len(nonsilence_low)):
            x_split = x_remove_edge[nonsilence_low[i][0]:nonsilence_low[i][1]]
            x_rec = np.append(x_rec, x_split)
    return x_rec

def get_local_mean_layer(ks):
    local_mean_layer = nn.Conv1d(1, 1, ks, 1, int((ks-1)/2))
    parameters = local_mean_layer.state_dict()
    bias = torch.zeros(parameters["bias"].shape)
    parameters["bias"] = bias
    weights = torch.ones(parameters["weight"].shape) / ks
    parameters["weight"] = weights
    local_mean_layer.load_state_dict(parameters)
    return local_mean_layer
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('linear') != -1:        
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('batchnorm') != -1:     
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
        
def plot_spectrogram(M, fig=None, subplot=(1,1,1), t=None, freq=None, title="", xlabel="", ylabel="", alpha=1, title_font=15):
    if fig==None:
        fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(subplot[0], subplot[1], subplot[2])
    if t==None:
        t = range(M.shape[1])
    if freq==None:
        freq = range(M.shape[0])
    ax.pcolormesh(t, freq, M, cmap = 'jet', alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=title_font)
    
def get_mel(x, fs, window_num=1024, stride_num=256, mel_num=80):
    if mel_num == None:
        mel_num = int(window_num/2)
    S = librosa.feature.melspectrogram(y=x, sr=fs, n_fft=window_num, hop_length=stride_num, n_mels=mel_num)
    S_log = 20*np.log10(S+1e-10)
    return S_log

def my_scheduler(lr, epoch, gamma):
    return lr * gamma ** epoch

class my_adam_waveform():
    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.m_t_1 = 0
        self.v_t_1 = 0
        self.eps = eps
    
    def update(self, data, loss, lr=0.001):
        grad = torch.autograd.grad(loss, data, retain_graph=True)[0]
        m_t = self.beta1*self.m_t_1 + (1-self.beta1)*grad
        v_t = self.beta2*self.v_t_1 + (1-self.beta2)*grad**2
        m = m_t / (1 - self.beta1)
        v = v_t / (1 - self.beta2)
        new = data - lr / (v**(1/2)+self.eps) * m
        self.m_t_1 = m_t
        self.v_t_1 = v_t
        return new
    
def transform(array, target=2**15):
    length = array.shape[0]
    if length > target:
        start = np.random.randint(length - target)
        new_array = min_max(array[start:start+target, :], mean0=True)
    elif length < target:
        zeros = np.zeros((target-length, 1))
        start = np.random.randint(target-length)
        new_array = np.concatenate([zeros[:start,:], min_max(array, mean0=True), zeros[start:,:]])
    elif length == target:
        new_array = min_max(array, mean0=True)
    return new_array

def transform_mel(array, target=160):
    length = array.shape[1]
    if length > target:
        start = np.random.randint(length - target)
        new_array = min_max(array[:, start:start+target], mean0=False)
    return new_array
    
########### for consistency regularization ########
class ToPIL(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return image_from_output(img.unsqueeze(0))[0]

    def __repr__(self):
        return self.__class__.__name__
    
augment = transforms.Compose([
    ToPIL(),
    transforms.RandomAffine(degrees=0, translate=(0.1,0)),
    transforms.ToTensor(),
])
    
    
def get_augmented_image(data, transform, mean0=True):
    for i in range(data.shape[0]):
        x = data[i]
        image = min_max_torch(transform(x), mean0=mean0)
        image = image.unsqueeze(0)[:,0:1,:,:]
        if i == 0:
            new = image
        else:
            new = torch.cat([new, image])
    return new
    
############ https://www.kaggle.com/grfiv4/plot-a-confusion-matrix #############
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()