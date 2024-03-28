# Move as You Say, Interact as You Can: Language-guided Human Motion Generation with Scene Affordance

<p align="left">
    <a href='https://arxiv.org/abs/2403.18036'>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
    &nbsp;
    <a href='https://afford-motion.github.io/static/pdfs/paper.pdf'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='Paper PDF'>
    </a>
    &nbsp;
    <a href='https://afford-motion.github.io/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=githubpages&logoColor=blue' alt='Project Page'>
    </a>
</p>

[Zan Wang](https://silvester.wang),
[Yixin Chen](https://yixchen.github.io/),
[Baoxiong Jia](https://buzz-beater.github.io/),
[Puhao Li](https://xiaoyao-li.github.io/),
[Jinlu Zhang](https://jinluzhang.site/),
[Jingze Zhang]()
[Tengyu Liu](http://tengyu.ai/),
[Yixin Zhu](https://yzhu.io/),
[Wei Liang](https://liangwei-bit.github.io/web/),
[Siyuan Huang](https://siyuanhuang.com/)

This repository is the official implementation of paper "Move as You Say, Interact as You Can:
Language-guided Human Motion Generation with Scene Affordance".

We introduce a novel two-stage framework that employs scene affordance as an intermediate representation, effectively linking 3D scene grounding and conditional motion generation.

[arXiv](https://arxiv.org/abs/2403.18036) | 
[Paper](https://afford-motion.github.io/static/pdfs/paper.pdf) | 
[Project](https://afford-motion.github.io/)

<div align=center>
<img src='./assets/teaser.png' width=60%>
</div>

## Abstract

Despite significant advancements in text-to-motion synthesis, generating language-guided human motion within 3D environments poses substantial challenges. These challenges stem primarily from (i) the absence of powerful generative models capable of jointly modeling natural language, 3D scenes, and human motion, and (ii) the generative models' intensive data requirements contrasted with the scarcity of comprehensive, high-quality, language-scene-motion datasets. To tackle these issues, we introduce a novel two-stage framework that employs scene affordance as an intermediate representation, effectively linking 3D scene grounding and conditional motion generation. Our framework comprises an Affordance Diffusion Model (ADM) for predicting explicit affordance map and an Affordance-to-Motion Diffusion Model (AMDM) for generating plausible human motions. By leveraging scene affordance maps, our method overcomes the difficulty in generating human motion under multimodal condition signals, especially when training with limited data lacking extensive language-scene-motion pairs. Our extensive experiments demonstrate that our approach consistently outperforms all baselines on established benchmarks, including HumanML3D and HUMANISE. Additionally, we validate our model's exceptional generalization capabilities on a specially curated evaluation set featuring previously unseen descriptions and scenes.

## Citation

If you find our project useful, please consider citing us:

```tex
@inproceedings{wang2024move,
  title={Move as You Say, Interact as You Can: Language-guided Human Motion Generation with Scene Affordance},
  author={Wang, Zan and Chen, Yixin and Jia, Baoxiong and Li, Puhao and Zhang, Jinlu and Zhang, Jingze and Liu, Tengyu and Zhu, Yixin and Liang, Wei and Huang, Siyuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

### License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.