# Activation_Maximization

This repository is to introduce some applications for [class-based Activation Maximization(AM)](https://arxiv.org/pdf/1312.6034.pdf) in audio domain.

---
## Introduction

Neural networks are predominant for various tasks including object detection, speech recognition, emotion detection, and so on. However, its process is, in general, not understandable for human beings. To understand how the models tackle the problems, some visualization techniques are invented such as feature visualizations. In this repository, I'm going to share the applications of Activation Maximization(AM) which is one of the feature visualization tactics.

Basically, in AM, the input data is optimized to the data that activates the selected neuron. It contains the filter of layers, the classification output, and so on. In our case, the output of the classifier is optimized to observe the result of being a certain class. That's why I called it class-based Activation Maximization, and this is mentioned in [this paper](https://arxiv.org/pdf/1312.6034.pdf). For further information, please visit [this excellent explanation for AM](https://distill.pub/2017/feature-visualization/)

In this experiment, I'm going to optimize the noise of GAN which is employed as a prior as shown below. As for the form of audio data, 2 types of audio features are employed, which are raw audio and mel-spectrogram. We're going to observe the differences between the data form and the structure of the models. What's more, Conditional GAN is also experimented to figure out the importance of being a certain emotion. Lastly, the biggest advantage of this idea is that it can be used as an enhancer of the model output. For example, in our case, the model was not able to generate audio which we expected, but this concept allowed the model to enhance its output specific to our purpose.

<img src="./data/images/system.png" width="800">

---
## Notebooks
This idea requires 2 models, including a classifier and a generator for GAN (or Conditional GAN). Some brief definition of the notebooks are as follows:
- `01_audio_emotion_classifier.ipynb`: emotion classification in audio domain
- `02_GAN_training.ipynb`: Training of GAN
- `03_GAN_audio_AM`: Activation Maximization in raw audio with GAN
- `04_mel_emotion_classifier.ipynb`: emotion classification in mel-spectrogram
- `05_GAN_mel_AM`: Activation Maximization in mel_spectrogram with GAN
- `06_result_GAN_AM`: Summary of the Activation Maximization in GAN
- `A_preprocessing_TESS_and_RAVDESS`: Brief introduction and Preprocessing of Datasets
- `A-download_Download_TESS_RAVDESS`: How to download TESS and RAVDESS datasets
- `B_WaveGlow_parameters`: Obtaining the parameters of WaveGlow
- `C_Emotion_Recognition-Inception`: emotion classification with Inception Model

---
## Results
---
<!-- ### ***Since I'm not allowed to post any audio data in README, I've posted the audio on [my blog](https://shinshoji01.hatenablog.com/entry/results_audioAM_prior).*** -->

### ***Since I'm not allowed to post any audio data in README, I've posted the audio on [my blog](https://shinshoji01.hatenablog.com/entry/results_audioAM_prior).***

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
## Docker
In this repository, we share the environment that you can run the notebooks.
1. Build the docker environment.
    - with GPU
      - `docker build --no-cache -f Docker/Dockerfile.gpu .`
    - without GPU
      - `docker build --no-cache -f Docker/Dockerfile.cpu .`
2. Check the \<IMAGE ID\> of the created image.
    - `docker images`
3. Run the docker environment
    - with GPU
      - `docker run --rm --gpus all -it -p 8080:8080 -e LOCAL_UID=$(id -u $USER) -e LOCAL_GID=$(id -g $USER) -v ~:/work <IMAGE ID> bash`
    - without GPU
      - `docker run --rm -it -p 8080:8080 -e LOCAL_UID=$(id -u $USER) -e LOCAL_GID=$(id -g $USER) -v ~:/work <IMAGE ID> bash`
4. Run the jupyter lab
    - `nohup jupyter lab --ip=0.0.0.0 --no-browser --allow-root --port 8080 --NotebookApp.token='' > nohup.out &`
5. Open the jupyter lab
    - Put http://localhost:8080/lab? to web browser.

---
## Installation of some apps

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
