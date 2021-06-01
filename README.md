# Handwritten Character Classification

Class project for DSE230. Initial plan is to build a distributed SVC model covering 39 categories of handwritten characters. Image dataset will be sourced from [Kaggle](https://www.kaggle.com/vaibhao/handwritten-characters), and consists of approximately 857000 files. Off the bat there looks to be rather significant class imbalance, so our resulting work will have to account for this (through model-level class weighting, dataset-level minority class generation, or some combination of both).

As for possible avenues of exploration, comparisons with any of the following would be a good start: self-trained neural network, cloud-hosted image recognition API, or a transfer learned ImageNet model. Alternatively, our model could be the backend for a web-based real time classification tool.
