# CS297-research-project-em

CS297 research project for Ethan Machado, San José State University.

Experiment 1 compares Qwen zero-shot/few-shot inference with supervised Tiny
Recursive Model (TRM) training for face-image degradation classification and
denoising routing decisions.

- [Local execution and experiment reference](experiment1/README.md)
- [Experiment 2: self-contained Kaggle GPU workflows](experiment2/README.md)
- [Colab Enterprise / GCP Workbench setup](experiment1/notebooks/colab_enterprise/README.md)
- [Pipeline and architecture](experiment1/PIPELINE.md)

Cloud notebooks include repository/dependency setup, download-if-missing dataset
preparation, CUDA execution, saved-run reports, and periodic Cloud Storage uploads.
Source and notebooks live in Git; datasets, model caches, credentials, and generated
artifacts stay outside version control.
