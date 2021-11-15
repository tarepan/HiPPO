<div align="center">

# HiPPO:  <!-- omit in toc -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][notebook]
[![Paper](http://img.shields.io/badge/paper-arxiv.2005.09409-B31B1B.svg)][paper]  

![HiPPO Framework](assets/hippo.png "HiPPO Framework")

</div>

Clone of official implmentation of HiPPO.  

## Quick Training
Jump to ☞ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][notebook], then Run. That's all!  

## How to Use

### 1. Install <!-- omit in toc -->
```bash
# pip install "torch>=1.4.0"      # Based on your PyTorch environment
pip install -r requirements.txt
```

### 2. Data & Preprocessing <!-- omit in toc -->
<!-- "Batteries Included" 😉   -->
<!-- Dataset class transparently downloads ZeroSpeech2019 corpus and preprocesses it for you.   -->

### 3. Training <!-- omit in toc -->
Launch experiments using `train.py`.

Pass in `dataset=<dataset>` to specify the dataset, whose default options are specified by the Hydra configs in `cfg/`. See for example `cfg/dataset/mnist.yaml`.

Pass in `model.cell=<cell>` to specify the RNN cell. Default model options can be found in the initializers in the model classes.

The following example command lines reproduce experiments in Sections 4.1 and 4.2 for the HiPPO-LegS model. The `model.cell` argument can be changed to any other model defined in `model/` (e.g. `lmu`, `lstm`, `gru`) for different types of RNN cells.

### Permuted MNIST

```
python train.py runner=pl runner.ntrials=5 dataset=mnist dataset.permute=True model.cell=legs model.cell_args.hidden_size=512 train.epochs=50 train.batch_size=100 train.lr=0.001
```

### CharacterTrajectories

See documentation in `datasets.uea.postprocess_data` for explanation of flags.

100Hz -> 200Hz:
```
python train.py runner=pl runner.ntrials=2 dataset=ct dataset.timestamp=False dataset.train_ts=1 dataset.eval_ts=1 dataset.train_hz=0.5 dataset.eval_hz=1 dataset.train_uniform=True dataset.eval_uniform=True model.cell=legs model.cell_args.hidden_size=256 train.epochs=100 train.batch_size=100 train.lr=0.001
```
Use `dataset.train_hz=1 dataset.eval_hz=0.5` instead for 200Hz->100Hz experiment.


Missing values upsample:
```
python train.py runner=pl runner.ntrials=3 dataset=ct dataset.timestamp=True dataset.train_ts=0.5 dataset.eval_ts=1 dataset.train_hz=1 dataset.eval_hz=1 dataset.train_uniform=False dataset.eval_uniform=False model.cell=tlsi model.cell_args.hidden_size=256 train.epochs=100 train.batch_size=100 train.lr=0.001
```
Use `dataset.train_ts=1 dataset.eval_ts=0.5` instead for downsample.

Note that the model cell is called tlsi (short for "timestamped linear scale invariant") to denote a HiPPO-LegS model that additionally uses the timestamps.


### HiPPO-LegS multiplication in C++
To compile:
```
cd csrc
python setup.py install
```
To test:
```
pytest tests/test_legs_extension.py
```
To benchmark:
```
python tests/test_legs_extension.py
```


## Original paper
[![Paper](http://img.shields.io/badge/paper-arxiv.2008.07669-B31B1B.svg)][paper]  
<!-- https://arxiv2bibtex.org/?q=2008.07669&format=bibtex -->
```
@misc{2008.07669,
Author = {Albert Gu and Tri Dao and Stefano Ermon and Atri Rudra and Christopher Re},
Title = {HiPPO: Recurrent Memory with Optimal Polynomial Projections},
Year = {2020},
Eprint = {arXiv:2008.07669},
}
```

[paper]:https://arxiv.org/abs/2008.07669
[notebook]:https://colab.research.google.com/github/tarepan/HiPPO/blob/master/HiPPO.ipynb
