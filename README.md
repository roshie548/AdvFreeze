# AdvFreeze - A novel adversarial training method

The objective of this project was to design and implement a robust TinyImageNet classifier that would perform well on a hidden test set, despite unknown test-time perturbation. To evaluate our models, we focused on two metrics: accuracy on a clean dataset and accuracy on an adversarially perturbed dataset, generated using Fast Gradient Sign Method (FGSM). We propose a new adversarial training process which greatly increases robustness without hurting clean accuracy. Our novel approach achieves this by selectively altering how batch norm layers operate while training on adversarial examples.

We trained an EfficientNet-B0 model with our novel process, achieving a clean validation accuracy of 75.85\% and an adversarial validation accuracy of 71.97\%, showing a +7.45\% clean improvement and a +21.64\% adversarial improvement over a vanilla adversarially-trained EfficientNet-B0. The problem of training a robust yet accurate image classifier is important as it will gives us more confidence to deploy CV models in the real world without worrying about possible detriments arising from unforeseen or malicious perturbations. Our work in this paper demonstrates a new way of training models to become more robust against malicious perturbations **without sacrificing accuracy on clean images**.

---

The full report is in report.pdf.
The rest of the repository includes all of our trained models, as well as running our novel method-trained model on a test set.

Note: You should be using Python 3 to run this code.

Running the code requires importing the EfficientNet-Pytorch package by lukemelas (this is already included in the requirements.txt, so it should be installed). (https://github.com/lukemelas/EfficientNet-PyTorch)

Usage for this the script is:
$ python test_submission.py {csv for input images}.

Training data is expected to be stored in ./data/tiny-imagenet-200/train/
