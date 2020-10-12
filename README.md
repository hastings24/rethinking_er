# Rethinking Experience Replay: a Bag of Tricks for Continual Learning
This code is based on our framework: [Mammoth - An Extendible Continual Learning Framework for Pytorch](https://github.com/aimagelab/mammoth).

To run experiments with the default arguments use `python ./utils/main.py --model=<MODEL> --dataset=<DATASET> --buffer_size=<MEM_BUFFER_SIZE> --load_best_args`.

Available models:

+ `sgd`: SGD with no countermeasure to catastrophic forgetting (lower bound)
+ `joint`: joint training on the whole dataset (upper bound - not continual)
+ `agem`: [A-GEM](https://arxiv.org/abs/1812.00420)
+ `gem`: [Gradient Episodic Memory for Continual Learning](https://arxiv.org/abs/1706.08840)
+ `hal`: [Hindisght Anchor Learning](https://arxiv.org/abs/2002.08165)
+ `iCaRL`: [Incremental Classifier and Representation Learning](https://arxiv.org/abs/1611.07725)
+ `er`: naive Experience Replay with no tricks
+ `er_tricks`: Experience Replay equipped with our proposed tricks

Available datasets:

+ `seq-fmnist`: Split Fashion-MNIST (5 tasks, 2 classes per task)
+ `seq-cifar10`: Split CIFAR-10 (5 tasks, 2 classes per task)
+ `seq-cifar100`: Split CIFAR-100 (10 tasks, 10 classes per task)
+ `seq-core50`: CORe50 dataset according to the SIT-NC protocol described [here](https://arxiv.org/abs/1806.08568)

Best args are provided for the following memory buffer sizes:

+ 200 exemplars
+ 500 exemplars
+ 1000 exemplars
