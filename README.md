<h1><u>About Hidden Emotion Detection Research</u></h1>

<b>Paper Title</b>: Hidden Emotion Detection using Multi-modal Signals

<b>Conference</b>: ACM CHI 2021

<b>Abstract:</b> In order to better understand human emotion, we should not recognize only superfcial emotions based on facial images, but also analyze so-called inner emotions by considering biological signals such as electroencephalogram (EEG). ... This paper defnes a new task to detect hidden emotions, i.e., emotions in a situation where only the EEG signal is activated without the image signal being activated, and proposes a method to effectively detect the hidden emotions. ... As a result, this study has upgraded the technology of deeply understanding inner emotions.

---
### Contents

- Basic setting of hidden emotion detection (HED)
<p align="center">
<img src="https://github.com/kdhht2334/Hidden_Emotion_Detection_using_MM_Signals/blob/main/pics/hed_pic_01.png" height="200", width="2254"/>
</p>

- Qualitative results (both positive and negative inner emotional states)
<p align="center">
<img src="https://github.com/kdhht2334/Hidden_Emotion_Detection_using_MM_Signals/blob/main/pics/hed_pic_02.png" height="410", width="3000"/>
</p>
<p align="center">
<img src="https://github.com/kdhht2334/Hidden_Emotion_Detection_using_MM_Signals/blob/main/pics/hed_pic_03.png" height="410", width="3000"/>
</p>



---
### Notes

We provide the HED database only for research purpose. The database consists of 246 video clips obtained from 23 experimental participants.

Specifically, HED database consists not only raw video clips but also EEG signals pre-processed by numpy array (`.npy`) for research convenience.

- For download raw video clips & Synced Visual/EEG npy 
  - [[LINK]](https://1drv.ms/u/s!AsMhRBCpiZ4ShcYgcfhDCHUcvZgHkA?e=6UvaY3)

---
### BibTeX

Please cite the paper if you choose to use HED database for your research.

```
@inproceedings{10.1145/3411763.3451721,
author = {Song, Byung Cheol and Kim, Dae Ha},
title = {Hidden Emotion Detection Using Multi-Modal Signals},
year = {2021},
doi = {10.1145/3411763.3451721},
booktitle = {Extended Abstracts of the 2021 CHI Conference on Human Factors in Computing Systems},
series = {CHI EA '21}
}
```