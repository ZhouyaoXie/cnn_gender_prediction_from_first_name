# cnn_gender_prediction_from_first_name

The goal is to train a binary classifier that outputs gender predictions given first names as inputs. After some literature research, I was pointed to two papers on character-level convolutional neural nets for text classification ([Zhang, Zhao, Lecun](https://arxiv.org/abs/1509.01626) and [Kim](https://arxiv.org/abs/1408.5882)). First names contain almost no semantic or syntactic information, which makes the task of inferring from first names quite different from understanding normal words. Since not much information is lost from viewing the text at character-level, character-level CNN seems appropriate for the task.

The training dataset is the [national name dataset ](https://www.ssa.gov/oact/babynames/limits.html) provided by U.S. Social Security Administration. I used the data from 1950 to 2018, which contains 517490 unique name-gender pairs (including ambiguous names). 2% of the data were randomly sampled to use as the testset. From the rest data, 20% were randomly selected as the validation set.

The neural network I implemented below slightly modified Zhang's design. It consists of one 128-filter, three 64-filter 1-D convolution layers, and two fully connected layers. Each convolution layer is followed by a max pooling layer, with pooling size equals to 3. I also used dropout in between the three dense layers to regularize. I used an Adam optimizer with learning rate 0.0005 to perform gradient descent.

The model attained an accuracy of **86.11%** on the testset.

I also referred to [this](https://github.com/mhjabreel/CharCnn_Keras), [this](https://github.com/Irvinglove/char-CNN-text-classification-tensorflow), and [this](https://github.com/BrambleXu/nlp-beginner-guide-keras/blob/master/char-level-cnn/char_cnn.py) github repos for the implementation of character-level CNN.
