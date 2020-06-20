# sciezkiOptyczne
Application of various machine learning methods for Quality of Transmission (QoT) estimation.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.


### Prerequisites
Install prerequisities:
- Anaconda (https://www.anaconda.com/products/individual#)

### Clone
Clone this repo to your local machine:
```
git clone git@github.com:wwrzesien/sciezkiOptyczne.git
```

### Setup
Create a virtual environment with a specific python version:
```
conda create --name myenv python=3.6
```
Activate an environment:
```
conda activate myenv
```
Install TensorFlow and Keras:
```
conda install tensorflow
conda install keras
```
Install Matplotlib:
```
conda install matplotlib
```

### Deployment
Activate an environment:
```
conda activate myenv
```
In `sciezkiOptyczne` run estimation:
```
python main.py
```

## Built With
- [TensorFlow](https://www.tensorflow.org/) - is a free and open-source software library for dataflow and differentiable programming across a range of tasks.
- [Keras](https://keras.io/) - is an open-source neural-network library written in Python.

## Authors
- **Wojciech Wrzesień** - [wwrzesien](https://github.com/wwrzesien)
- **Marcin Gajewski** - [marcingajewski14](https://github.com/marcingajewski14)
