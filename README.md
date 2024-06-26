
# ⚽️Explainable Video Assistant Referee System⚽️ 


| Paper | Demo Clips | Dataset | Model | Training | Offline Demo | Examples |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| [![paper](https://img.shields.io/badge/Paper-<COLOR>.svg)](https://arxiv.org/abs/2404.06332) | [![DemoClip-1](https://img.shields.io/badge/-DemoClip1-blue)](https://www.youtube.com/watch?v=IJAkf94lPO8) [![DemoClip-2](https://img.shields.io/badge/-DemoClip2-yellow)](https://www.youtube.com/watch?v=SEdUjrLMBik) [![DemoClip-3](https://img.shields.io/badge/-DemoClip3-green)](https://www.youtube.com/watch?v=Nkk125MvkJ0) [![DemoClip-4](https://img.shields.io/badge/-DemoClip4-violet)](https://www.youtube.com/watch?v=yG6qV9jsAJI) | [SoccerNet-XFoul](#SoccerNet-XFoul) | [X-VARS](#X-VARS) | [Training](#Training) |  [Offline Demo](https://github.com/heldJan/X-VARS/blob/main/X-VARS/demo.md) |  [Examples](#Examples) |


This repository contains: 
- **SoccerNet-XFoul**, a novel dataset consisting of more than **22k video-question-answer** triplets annotated by over 70 experienced football referees. 🚀
- **X-VARS**, a new vision language model that can **perform multiple multi-modal tasks** such as visual captioning, question-answering, video action recognition, and can generate explanations of its decisions on-par with human level. 🤖
- The code to run an **offline demo on your laptop**. 💻

## 📢 NEWS 📢

The **SoccerNet-XFoul dataset** and the **pretrained weights** will be available soon! 🔥🔥

## Installation

We recommend setting up a conda environment for the project:

```
conda create --name=xvars python=3.10
conda activate xvars

git clone https://github.com/heldJan/X-VARS.git
cd X-VARS
pip install -r requirements.txt
```

## SoccerNet-XFoul

The SoccerNet-XFoul dataset consists of 22k video-question-answer pairs annotated by more than 70 experienced referees. 
Due to the subjectivity in refereeing, we gathered multiple answers for the same action, rather than collecting a single decision and explanation for each question. In the end, for each action, we have, on average, $1.5$ answers for the same question.

Follow the [link](https://pypi.org/project/SoccerNet/) to easily download the SoccerNet pip package.

If you want to download the video clips, you will need to fill a [NDA](https://docs.google.com/forms/d/e/1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform) to get the password.

Then use the API to downlaod the data:

```
from SoccerNet.Downloader import SoccerNetDownloader as SNdl
mySNdl = SNdl(LocalDirectory="path/to/SoccerNet")
mySNdl.downloadDataTask(task="mvfouls", split=["train","valid","test","challenge"], password="enter password")
```
To obtain the data in 720p, add version = "720p" to the input arguments. If you face issues extracting data from the train_720p.zip folder, the error may come from using the default unzip extractor. Using the app "The Unarchiver" should enable you to unzip it successfully.

The annotations will be available soon! Stay tuned🔥

![My Image](Images/dataset_example.png)


## X-VARS

X-VARS is a visual language model based on a **fine-tuned CLIP visual encoder** to extract **spatio-temporal video features** and to obtain **multi-task predictions** regarding the type and severity of fouls. The **linear layer** connects the vision encoder to the language model by projection the video features in the text embedding dimension. We input the **projected spatio-temporal features** alongside **the text predictions** obtained by the two classification heads (for the task of determining the type of foul and the task of determining if it is a foul and the corresponding severity) into the Vicuna-v1.1 model, initialized with weights from LLaVA.
![My Image](Images/architecture.png)


## Training 

We propose a two-stage training approach. **The first stage** fine-tunes CLIP on a multi-task classification to learn prior knowledge about football and refereeing. **The second stage** consists in fine-tuning the projection layer and several layers of the LLM to enhance the model's generation abilities in the sport-specific domain. 

To replicate the training, check out [Training](https://github.com/heldJan/X-VARS/blob/main/X-VARS/training.md)

More information are provided in our [paper](https://arxiv.org/abs/2404.06332).


## Examples
![My Image](Images/holding_no_card_main.png)
![My Image](Images/tackling_yellow_main.png)
![My Image](Images/dive.png)
![My Image](Images/tackling.png)

## Acknowledgements

 - [VARS:](https://github.com/SoccerNet/sn-mvfoul) The first multi-task classification model for predicting if it is a foul or not and the corresponding severity.
 - [Video-ChatGPT:](https://github.com/mbzuai-oryx/Video-ChatGPT) A vision and language model used as a foundation model for X-VARS


If you're using X-VARS in your research or application, please cite our paper: 
```bibtex
    @article{Held2024XVARS-arxiv,
        title = {X-{VARS}: Introducing Explainability in Football Refereeing with Multi-Modal Large Language Model},
        author = {Held, Jan and Itani, Hani and Cioppa, Anthony and Giancola, Silvio and Ghanem, Bernard and Van Droogenbroeck, Marc},
        journal = arxiv,
        volume = {abs/2404.06332},
        year = {2024},
        publisher = {arXiv},
        eprint = {2404.06332},
        keywords = {},
        eprinttype = {arXiv},
        doi = {10.48550/arXiv.2404.06332},
        url = {https://doi.org/10.48550/arXiv.2404.06332}
}

}
```
## Authors 

- [Jan Held](https://www.linkedin.com/in/heldjan/)
- [Hani Itani](https://www.linkedin.com/in/hani-itani-613b01111/)
- [Anthony Cioppa](https://www.linkedin.com/in/anthony-cioppa-65a779128/)
- [Silvio Gincola](https://www.linkedin.com/in/silvio-giancola/)
- [Bernard Ghanem](https://www.linkedin.com/in/bernardghanem/)
- [Marc Van Droogenbroeck](https://scholar.google.com/citations?user=IQm8ljgAAAAJ&hl=en)

[<img src="Images/uliege.png" width="150" height="70">](https://www.ival-mbzuai.com)
[<img src="Images/kaust.png" width="150" height="80">](https://github.com/mbzuai-oryx)
