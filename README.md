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

For a more numerical explanation, this image from Andrej Karpathy's [blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/ "Unreasonable Effectiveness of RNNs") is quite intuitive, particularly in the context of character prediction.

![alt text](https://github.com/SolomonZA/AMLR/blob/master/img/rnn_unrolled_1.jpeg "Unrolled RNN")

The rectangles represent vectors and the arrows weight matrices. Red arrows map from the input layer to the hidden layer, green arrows show the mapping from the previous hidden layer (state) to the current, and the blue arrows depict the mapping from the hidden layer to the output layer.

The start of the sequence is "h" at "time step" **t**, the [1 0 0 0] vector representing this first character is mapped to the hidden layer resulting in a certain activation, and then to both 1) the "next" time step's hidden layer, and 2)the output layer, with different weight matrices, the latter's mapping showing a high probability of "e" being the *next character*.

The second character of the sequence is indeed "e" at time step **t+1**, and this follows a similar process as with "h". However, at the hidden layer, the output of the previous time step's hidden layer is now taken into account at this time step **t+1**.

The hidden layer then maps these activations to the output layer. The output vector, indicating a strong probability for "l", can be interpreted roughly as the following:

*Conditional on the previous (time step **t**) character having been an "h", the current character input (time step **t+1**) being an "e" suggests a high probability that the output, or following character (time step **t+2**) will be an "l".