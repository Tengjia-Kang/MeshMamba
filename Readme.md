# Mesh Mamba: A Unified State Space Model for Saliency Prediction in Non-Textured and Textured Meshes (CVPR 2025)
This is the source code for the paper "[Mesh Mamba: A Unified State Space Model for Saliency Prediction in Non-Textured and Textured Meshes](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Mesh_Mamba_A_Unified_State_Space_Model_for_Saliency_Prediction_CVPR_2025_paper.html)".

[Kaiwei Zhang](https://github.com/kaviezhang), [Dandan Zhu](), [Xiongkuo Min](https://scholar.google.com/citations?user=91sjuWIAAAAJ&hl=en&oi=ao), and [Guangtao Zhai](https://scholar.google.com/citations?user=E6zbSYgAAAAJ&hl=en&oi=ao)

<hr />

> **Abstract:** *Mesh saliency enhances the adaptability of 3D vision by identifying and emphasizing regions that naturally attract visual attention. To investigate the interaction between geometric structure and texture in shaping visual attention, we establish a comprehensive mesh saliency dataset, which is the first to systematically capture the differences in saliency distribution under both textured and non-textured visual conditions. Furthermore, we introduce mesh Mamba, a unified saliency prediction model based on a state space model (SSM), designed to adapt across various mesh types. Mesh Mamba effectively analyzes the geometric structure of the mesh while seamlessly incorporating texture features into the topological framework, ensuring coherence throughout appearance-enhanced modeling. More importantly, by subgraph embedding and a bidirectional SSM, the model enables global context modeling for both local geometry and texture, preserving the topological structure and improving the understanding of visual details and structural complexity. Through extensive theoretical and empirical validation, our model not only improves performance across various mesh types but also demonstrates high scalability and versatility, particularly through cross validations of various visual features.* 
<hr />

## 📋 TODO
- [x] Release the training and evaluation code
- [x] Release the pretrained weights
- [x] Release the dataset.

## 🎒 Data preparation
[Dataset](https://drive.google.com/drive/folders/1he9DBx4uRoDg-Fx_2Ec769o4JNc8fqIS?usp=sharing)
Then run get_neighbor_ringn.py to prepare the data.

## 💫 Start training
- Environment: Python 3.8.18 and requirement.txt
- Change the base_root in get_neighbor_ringn.py to prepare the data.
- Change the data_root in config/MeshSaliency.yaml to your dataset path.
- Run the train_mesh_Mamba.py

## ❤️ Acknowledgement
This project is based on Mamba ([paper](https://arxiv.org/abs/2312.00752), [code](https://github.com/state-spaces/mamba)), Vision Mamba ([paper](https://icml.cc/virtual/2024/poster/33768), [code](https://github.com/hustvl/Vim)), Mamba3D ([paper](https://dl.acm.org/doi/abs/10.1145/3664647.3681173), [code](https://github.com/xhanxu/Mamba3D)). Thanks for their wonderful works.

## 📚 Citation
If you use MeshMamba, please consider citing:
```bibtex
@inproceedings{zhang2025mesh,
  title={Mesh Mamba: A Unified State Space Model for Saliency Prediction in Non-Textured and Textured Meshes},
  author={Zhang, Kaiwei and Zhu, Dandan and Min, Xiongkuo and Zhai, Guangtao},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={16219--16228},
  year={2025}
}
```
