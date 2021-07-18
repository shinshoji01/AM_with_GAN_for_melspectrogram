# Activation_Maximization

This repository is to introduce some applications for [class-based Activation Maximization(AM)](https://arxiv.org/pdf/1312.6034.pdf) in audio domain.

---
## Introduction

Neural networks are predominant for various tasks including object detection, speech recognition, emotion detection, and so on. However, its process is, in general, not understandable for human beings. To understand how the models tackle the problems, some visualization techniques are invented such as feature visualizations. In this repository, I'm going to share the applications of Activation Maximization(AM) which is one of the feature visualization tactics.

Basically, in AM, the input data is optimized to the data that activates the selected neuron. it contains the filter of layers, the classification output, and so on. In our case, the output of the classifier is optimized to observe the result of being a certain class. That's why I called it class-based Activation Maximization, and this is mentioned in [this paper](https://arxiv.org/pdf/1312.6034.pdf). For further information, please visit [this excellent explanation for AM](https://distill.pub/2017/feature-visualization/)

<img width="500" alt="Screen Shot 2020-08-11 at 15 27 54" src="https://user-images.githubusercontent.com/28431328/89864658-43d23780-dbe7-11ea-9318-9f705dbb02af.png">

In addition, there are some papers and articles I took ideas from:
- [Understanding Neural Networks via Feature Visualization: A survey](https://arxiv.org/abs/1904.08939)
- [Synthesizing the preferred inputs for neurons in
neural networks via deep generator networks](https://papers.nips.cc/paper/6519-synthesizing-the-preferred-inputs-for-neurons-in-neural-networks-via-deep-generator-networks.pdf)
- [How to visualize convolutional features in 40 lines of code](https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030)

---
## Applications
As I said, in this repo, I'm going to share some applications of class-based AM. These include:
- Mask vector optimization in the audio domain (in progress)
- voice optimization with a prior (done)

### mask vector optimization in the audio domain
In this experiment, Instead of updating the input data directly, I'm going to optimize the mask vector which is multiplied by the input data, as shown below. This will allow us to observe the importance of intensity or intonation of the audio in order for the model to understand the emotion.

please visit `audio_mask_vector_optimization/README.md` for the detail and the result.

### voice optimization with a prior
In this experiment, I'll optimize the noise of GAN which is employed as a prior. As for the form of audio data, 2 types of audio features are employed, which are raw audio and mel-spectrogram. We're going to observe the differences between the data form and the structure of the models. What's more, Conditional GAN is also experimented to figure out the importance of being a certain emotion.

please visit `GAN/README.md` for the detail and the result.

---
## Installation of some apps
The basic pip requirements are noted in `data/requrements.txt`. In addition to it,  please install some apps mentioned below to try my notebooks. 

**Git LFS (large file storage)**

Since this repository contains the parameters of the models. I used Git LFS to store a large file. The codes below are the recipe for this.

```bash
brew update
brew install git-lfs
```
- then, navigate to this repository.
```bash
git lfs install
git lfs fetch --all
git lfs pull
```

---
## Coming soon
Some are not explained which include:
- explanations of some functions and models.

---
## Contact
Feel free to contact me if you have any questions (<s-inoue-tgz@eagle.sophia.ac.jp>).
