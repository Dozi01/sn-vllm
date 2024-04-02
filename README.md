
# 🚀 Explainable Video Assistant Referee System - X_VARS 🚀

This repository contains: 
- SoccerNet-XFoul, a novel dataset consisting of more than 22k video-question-answer triplets annotated by over 70 experienced football referees. 
- X-VARS, a new vision language model that can perform multiple multi-modal tasks such as visual captioning, question-answering, video action recognition, and can generate explanations of its decisions on-par with human level.
- The code to run an offline demo.


# SoccerNet-XFoul

The SoccerNet-XFoul dataset consists of 22k video-question-answer pairs annotated by more than 70 experienced referees. 
Due to the subjectivity in refereeing, we gathered multiple answers for the same action, rather than collecting a single decision and explanation for each question. In the end, for each action, we have, on average, $1.5$ answers for the same question.

![My Image](Images/dataset_example.png)


## Acknowledgements

 - [VARS:](https://github.com/SoccerNet/sn-mvfoul) The first multi-task classification model for predicting if it is a foul or not and the corresponding severity.
 - [Video-ChatGPT:](https://github.com/mbzuai-oryx/Video-ChatGPT) A vision and language model used as a foundation model for X-VARS


If you're using X-VARS in your research or applications, please cite our paper: 




![Logo](uliege.png)
![Logo](kaust.png)
