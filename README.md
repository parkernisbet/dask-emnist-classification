# Dask EMNIST Classification

Initial plan is to build a distributed boosted decision trees model (LightGBM) covering 39 categories of handwritten characters. The image dataset will be sourced from [Kaggle](https://www.kaggle.com/vaibhao/handwritten-characters), and consists of approximately 857000 image files. Off the bat there looks to be rather significant class imbalance, so our working set of files will need to be reduced to even out these ratios.

The project directory contains four files, this README.md, a Python script 'main.py', a Jupyter Notebook 'main.ipynb', and a 'requirements.txt' file. The project was developed predominantly in the .py file, then later converted to a Jupyter notebook via Jupytext.

The notebook is self-contained, and will download all necessary files once run. Some steps are rather time consuming, particularly the unzipped of the image directories, dimension reduction, and hyperparameter tuning. The notebook won't automatically install packages required to run, instead a 'requirements.txt' file is provided to streamline installation and setup of your Python environment.

Execution was completed on a AMD Ryzen 3700X and 32 GB of RAM, so all times recorded in the notebook are a reflection of that. Some steps are memory intensive and so workers with less than 6GB of RAM, or systems with less than 16GB of RAM total will likely not be able to run this notebook.
