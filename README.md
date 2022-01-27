# Rethinking Learning Dynamics in RL using Adversarial Networks

### Getting started

To avoid any conflict with your existing Python setup, it is suggested to work in a virtual environment with [`virtualenv`](https://docs.python-guide.org/dev/virtualenvs/). To install `virtualenv`:
```bash
pip install --upgrade virtualenv
```
Create a virtual environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).
```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Datasets

In this work, we experiment with few datasets such as PointMass environment, 2-D Navigation environment and Metaworld.
PointMass environment can be used upon installation of garage. 2-D Navigation environment does not require any installation, and can be used as is. Metaworld environment has been used as prescribed in their paper:

```
@inproceedings{yu2019meta,
  title={Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning},
  author={Tianhe Yu and Deirdre Quillen and Zhanpeng He and Ryan Julian and Karol Hausman and Chelsea Finn and Sergey Levine},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2019}
  eprint={1910.10897},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
  url={https://arxiv.org/abs/1910.10897}
}
```


Note that, as long as the packages from the requirements are installed, you could run all of our experiments without any additional hassle!

### Training & Testing

We have carefully organized our codes under [`scripts`](scripts).

The TE-PPO model can be trained on all seeds in a parallel fashion as follows:
```bash
sbatch scripts/train_te_ppo_<dataset>.sh
```
Similarly, our ATE-PPO models can be tested on a fixed set of tasks in a parallel fashion as follows:
```bash
sbatch scripts/train_ate_ppo_<dataset>.sh
```

All our codes were run with 1 GPUs (TitanRTX or A100). Furthermore, all our codes have been optimized to be run on less than 30Gb RAM, including our experiments on Meta-World!

# Paper Citation

If you find our codes useful, do consider citing our paper:
```
Add arxiv link here
```

# References

Our repository makes use of various open-source codes. Most of which have been documented at Garage If you find the respective codes useful, do cite their respective papers:

```
@misc{garage,
 author = {The garage contributors},
 title = {Garage: A toolkit for reproducible reinforcement learning research},
 year = {2019},
 publisher = {GitHub},
 journal = {GitHub repository},
 howpublished = {\url{https://github.com/rlworkgroup/garage}},
 commit = {be070842071f736eb24f28e4b902a9f144f5c97b}
}
```
