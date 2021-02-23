# Housing Prices Assignment

This assignment aims to build a deep artificial neural network fora classification problem using the California Housing Prices dataset. The problem consists of estimating the approximate location of housing blocks. The approximate location is represented with a discrete variable called ocean_proximity which may have one of four possible values: NEAR BAY, 1H OCEAN, INLAND, and NEAR OCEAN.

## Implementation

### Preprocesing

The data provided is split into input features and labels. The instances corresponding to the value "ISLAND" for the ocean_proximity variable have been removed because there is not enough data forthe learning process, as well as instances with missing data.

Regarding the transformations within the data, the input features features have been normalized,which means they have similar orders of magnitude, specifically in the range [−1,1]. This can be done with the method MinMaxScaler in the package preprocessing of sklearn. Besides, the labels recieved have been one-hot encoded, so no further preprocessing was required.

Finally, the data was shuffled and divided manually according to the following percentages:

- Train set: 80%, being a total of 16342 rows.
- Validation set: 10%, being a total of 2044 rows.
- Test set: 10%, being a total of 2042 rows.

## Architecture

Our ffNN structure for the dataset is based on three dense layers with decreasing number of nodes and dropout between each layer. In order to choose the number of nodes, dropout parameters, learning rates and optimizers we made use of hyperparameter tuning. Here are some of the tested components:

- Dense layers sizes: As we previously commented, we experimented with different sizes per layer, inthe range of [32, 64, 128, 256, 512, 700]
- Dropout: It was used for reduce overfitting in our neural network by preventing complex co-adaptation on training data. The values we tested with were: [0.05, 0.1, 0.15, 0.2, 0.25].
- Optimizers: Adam, Adamax, Nadam and SGD.
- Activation Functions: Relu and LeakyRelu.
- Learning Rate: 1e−03, 1e−04.

## Results

The detailed results can be found on the [assignment report](housing_prices_report.pdf) and the performed hypertuning in the [csv file](housing_prices_hypertuning.csv). The best results obtained were:

- Train Accuracy and Loss: 96.73% and 0.0844.
- Validation Accuracy and Loss: 95.84% and 0.11.
- Test Accuracy and Loss: 97.11% and 0.0836. Elapsed time: 0.1172.

## Authors

- Student Name 1: Stefano Baggetto
- Student Name 2: Giorgio Segalla
- Student Name 3: Angel Igareta ([angel@igareta.com](angel@igareta.com))
