# voice optimization with a prior

---
## Introduction
As I mentioned in `Activation_Maximization/README.md`, Activation Maximization(AM) is feature visualization which is done by optimizing the input vector in order to activate the neuron, especially the class output, as illustrated below.

<!-- <img src="./data/images/AM.png" width="400"> -->
<img width="400" alt="AM" src="https://user-images.githubusercontent.com/28431328/106088313-17947e00-6169-11eb-8ab6-5af032d66bf6.png">


In this experiment, I'm going to optimize the noise of GAN which is employed as a prior as shown below. As for the form of audio data, 2 types of audio features are employed, which are raw audio and mel-spectrogram. We're going to observe the differences between the data form and the structure of the models. What's more, Conditional GAN is also experimented to figure out the importance of being a certain emotion. Lastly, the biggest advantage of this idea is that it can be used as an enhancer of the model output. For example, in our case, the model was not able to generate audio which we expected, but this concept allowed the model to enhance its output specific to our purpose.

<!-- <img src="./data/images/AM_with_prior.png" width="600"> -->
<img width="600" alt="AM_with_prior" src="https://user-images.githubusercontent.com/28431328/106088387-36931000-6169-11eb-8a71-cd9f8253588e.png">

---
## Notebooks
This idea requires 2 models, including a classifier and a generator for GAN (or Conditional GAN). Some brief definition of the notebooks are as follows:
- `01_audio_emotion_classifier.ipynb`: emotion classification in audio domain
- `02_GAN_training.ipynb`: Training of GAN
- `02-A_cGAN_training.ipynb`: Training of Conditional GAN
- `03_GAN_audio_AM`: Activation Maximization in raw audio with GAN
- `03-A_cGAN_audio_AM`: Activation Maximization in raw audio with Conditional GAN
- `04_mel_emotion_classifier.ipynb`: emotion classification in mel-spectrogram
- `05_GAN_mel_AM`: Activation Maximization in mel_spectrogram with GAN
- `05-A_cGAN_mel_AM`: Activation Maximization in mel_spectrogram with Conditional GAN
- `06_result_GAN_AM`: Summary of the Activation Maximization in GAN
- `06-A_result_cGAN_AM`: Summary of the Activation Maximization in Conditional GAN
- `A_preprocessing_TESS_and_RAVDESS`: Brief introduction and Preprocessing of Datasets
- `B_WaveGlow_parameters`: Obtaining the parameters of WaveGlow

---
## Results
---
<!-- ### ***Since I'm not allowed to post any audio data in README, I've posted the audio on [my blog](https://shinshoji01.hatenablog.com/entry/results_audioAM_prior).*** -->

### ***Since I'm not allowed to post any audio data in README, I've posted the audio on [my blog](https://blind-review.hatenablog.com/entry/audio_AM_result).***

---
Please visit `GAN/notebook/06_result_GAN_AM.ipynb` or `GAN/notebook/06-A_result_cGAN_AM.ipynb` for additional results and discussions.

**neutral**

<!-- <img src="./data/results/images/GAN_models_neutral_sample_4.png" width="900"> -->
![GAN_models_neutral_sample_4](https://user-images.githubusercontent.com/28431328/106088166-c5ebf380-6168-11eb-9059-66b99cc68716.png)


<!-- ![neutral](./data/gif/neutral.gif) -->
![neutral](https://user-images.githubusercontent.com/28431328/106087788-0f880e80-6168-11eb-98f3-156b1279cc65.gif)


**sad**

<!-- <img src="./data/results/images/GAN_models_sad_sample_0.png" width="900"> -->
![GAN_models_sad_sample_0](https://user-images.githubusercontent.com/28431328/106088199-d8fec380-6168-11eb-8bff-5dae2fe9c4c1.png)

<!-- ![sad](./data/gif/sad.gif) -->
![sad](https://user-images.githubusercontent.com/28431328/106087817-20388480-6168-11eb-9f9b-8e88090c9a6d.gif)

**angry**

<!-- <img src="./data/results/images/GAN_models_angry_sample_0.png" width="900"> -->
![GAN_models_angry_sample_0](https://user-images.githubusercontent.com/28431328/106088206-db611d80-6168-11eb-92b1-25e21b26fb9b.png)

<!-- ![angry](./data/gif/angry.gif) -->
![angry](https://user-images.githubusercontent.com/28431328/106087395-41e53c00-6167-11eb-9598-7e12c827ca97.gif)

**happy**

<!-- <img src="./data/results/images/GAN_models_happy_sample_3.png" width="900"> -->
![GAN_models_happy_sample_3](https://user-images.githubusercontent.com/28431328/106088211-de5c0e00-6168-11eb-86d7-71e29534a9b9.png)

<!-- ![happy](./data/gif/happy.gif) -->
![happy](https://user-images.githubusercontent.com/28431328/106087751-fb441180-6167-11eb-93c1-a0cbe9237ffd.gif)

### Further Research
- AM while fixing the text information.
- employ a model which is capable of adding emotion to audio, and use it as a prior.

---
## Installation of some apps
The basic pip requirements are noted in `data/requrements.txt`. In addition to it,  please install some apps in reference to `Activation_Maximization/README.md`. 
- Git LFS (large file storage)
- Plotly

---
## Coming soon
Some are not explained which include:
- explanations of some functions and models.

---
## Contact
Feel free to contact me if you have any questions (<s-inoue-tgz@eagle.sophia.ac.jp>).
