# Feed-Forward Neural Network

In this practical exercise the goal is to implement a Feed-Forward Neural Network (ffNN), instead of the common model of Convolutional Neural Network (CNN), for solving two image classification problems: [German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/gtsrb_news.html) and [Cifar-100](https://web.stanford.edu/~hastie/CASI_files/DATA/cifar-100.html).

## Implementation

The base implementation in Python with the tensorflow library is given and the aim is to decide on the following factors:

- Number of layers and number of units in each layer.
- Optimization parameters and algorithms to train the net.
- When to stop training according to the evolution of training during the optimization.
- Regularization.

The preprocessing performed consists of the following steps:

- Normalize the images between [0 , 1].
- Convert class vector to binary class matrix.
- Convert pictures to grayscale.
- Gaussian blur.

## Results

The best configuration found for the [German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/gtsrb_news.html) achieved a 92.5\% of training and 84.13\% in validation. The architecture consisted of:

- Three layers with decreasing number of nodes.
- Two layer with decreasing number of nodes 300-150 neurons.
- Two layer with constant number of nodes.

The best configuration found for the [Cifar-100](https://web.stanford.edu/~hastie/CASI_files/DATA/cifar-100.html) achieved a 33.16\% of training and 27.88\% in validation. The architecture consisted of:

- Three layers with decreasing number of nodes 1024 - 768 - 512 neurons.
- Two layers with decreasing number of nodes.
- Two layers with constant number of nodes.

More details can be found on the [presentation](feed_forward_neural_network_presentation.pdf).

## Authors

- Student Name 1: Stefano Baggetto
- Student Name 2: Giorgio Segalla
- Student Name 3: Angel Igareta ([angel@igareta.com](angel@igareta.com))
