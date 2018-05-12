# AMLR
Applied Machine Learning Research
---

This is a repo for developing and applying machine learning models.

The models are primarily neural networks.

The domain of these models are natural language processing tasks, such as next-character prediction.

In addition, some work on machine learning interpretability will also be included in this repo.

# Pure Python (NumPy) Recurrent Neural Network
---

The Pure Python (NumPy) Recurrent Neural Network (from now on Pure RNN) is a "fully recurrent" RNN.

The primary distinction between RNNs and traditional feedforward neural networks, is that RNNs exhibit temporal dynamic behaviour. It generates a **hidden state** which contains information about all prior inputs. This makes RNNs useful for modelling sequential processes such as **speech recognition** or **text prediction**.

It is possible to depict the RNN as a deep feedforward network. This conceptual exercise is known as "unrolling" or "unfolding" and it involves each time step explicitly, illustrating that the hidden state at time step *t* is a function of both the inputs *x_t* at time step *t* and the hidden state *h_{t-1}* of the previous time step.

The [image](https://en.wikipedia.org/wiki/Recurrent_neural_network "RNN - Wikipedia") below graphically depicts the process.

![alt text](https://github.com/SolomonZA/AMLR/blob/master/img/rnn_unrolled.png "Logo Title Text 1")

For a more numerical explanation, this image from Andrej Karpathy's [blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/ "Unreasonable Effectiveness of RNNs") is quite intuitive.

![alt text](https://github.com/SolomonZA/AMLR/blob/master/img/rnn_unrolled_1.jpeg "Unrolled RNN")