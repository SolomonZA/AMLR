# AMLR
Applied Machine Learning Research
---

This is a repo for developing and applying machine learning models.

The models are primarily neural networks.

The domain of these models are natural language processing tasks, such as next-character prediction.

In addition, some work on machine learning interpretability will also be included in this repo.

# Pure Python (NumPy) Recurrent Neural Network
---

The Pure Python (NumPy) Recurrent Neural Network (from now on Pure RNN) is a "fully recurrent" RNN. It uses only the NumPy library and is thus a highly transaprent "whitebox" implementation.

The primary distinction between RNNs and traditional feedforward neural networks, is that RNNs exhibit temporal dynamic behaviour. It generates a **hidden state** which contains information about all prior inputs. This makes RNNs useful for modelling sequential processes such as **speech recognition** or **text prediction**.

It is possible to depict the RNN as a deep feedforward network. This conceptual exercise is known as "unrolling" or "unfolding" and it involves showing each time step explicitly, illustrating that the hidden state at time step *t* is a function of both the inputs *x_t* at time step *t* and the hidden state *h_{t-1}* of the previous time step.

The [image](https://en.wikipedia.org/wiki/Recurrent_neural_network "RNN - Wikipedia") below graphically depicts the process.

![alt text](https://github.com/SolomonZA/AMLR/blob/master/img/rnn_unrolled.png "Logo Title Text 1")

For a more numerical explanation, this image from Andrej Karpathy's [blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/ "Unreasonable Effectiveness of RNNs") is quite intuitive, particularly in the context of character prediction.

![alt text](https://github.com/SolomonZA/AMLR/blob/master/img/rnn_unrolled_1.jpeg "Unrolled RNN")

The rectangles represent vectors and the arrows weight matrices. Red arrows map from the input layer to the hidden layer, green arrows show the mapping from the previous hidden layer (state) to the current, and the blue arrows depict the mapping from the hidden layer to the output layer.

The start of the sequence is "h" at "time step" **t**, the [1 0 0 0] vector representing this first character is mapped to the hidden layer resulting in a certain activation, and then to both 1) the "next" time step's hidden layer, and 2) the output layer, with different weight matrices, the latter's mapping showing a high probability of "e" being the *next character*.

The second character of the sequence is indeed "e" at time step **t+1**, and this follows a similar process as with "h". However, at the hidden layer, the output of the previous time step's hidden layer is now taken into account at this time step **t+1**.

The hidden layer then maps these activations to the output layer. The output vector, indicating a strong probability for "l", can be interpreted roughly as the following:

*Conditional on the previous (time step **t**) character having been an "h", the current character input (time step **t+1**) being an "e" suggests a high probability that the output, or following character (time step **t+2**) will be an "l".*

# Long short-term Memory Keras Implementation

Keras is a user-friendly and modular neural network library written in Python. It can use as backend various deep learning architectures, including TensorFlow an Theano. It is very useful for rapid prototyping of neural network models, particularly if your goal is to quickly determine whether your dataset or problem is amenable to a neural network based model.

In contrast to the Pure NumPy RNN, this implementation is more of a "black box", as it utilises several built in functions and classes from the Keras toolkit, to define the various components of the neural network. This includes the various layers, the LSTM nodes, and the model fitting/training process as well. This will therefore not serve the purpose of fundamentally understanding the nuances of an LSTM compared to either feedforward networks or RNNs - a later guide will. 

The Long short-term memory (LSTM) unit is a component of a recurrent neural network. It's primary contribution to RNNs is that it expands an RNN's capability to model input sequences in which there may be significant lags between important events. It does so by addressing the "vanishing gradient" problem.

A typical example cited, including by Martin Gorner, is of an input  paragraph of text that contains the phrase "I was born in Paris, France," at the beginning, then continues on with various pieces of unrelated (unimportant) information pertaining to the author and others, including "My dog's name is Bingo" as the penultimate sentence. The paragraph then ends with "I speak..." and the task at hand is for the model to predict the word "French", based on the author's first words indicating his French birthplace.

The problem arises because the important phrase, "I was born in Paris, France" is much further away from the target, than is "My dog's name is Bingo", which is a much less important phrase. The gradients which would have connected this important phrase to the target, are far back and have been reduced to near-0 values through training iterations.

This diagram below from [colah's blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/ "LSTM Module") illustrates the LSTM module.

<img src="https://github.com/SolomonZA/AMLR/blob/master/img/lstm.png" alt="LSTM Module" width="745" height ="280" />

The working of this module is described by the following set of equations:
<p align = 'center'>
	<img src="https://github.com/SolomonZA/AMLR/blob/master/img/lstm_eq.png" alt="LSTM Equations"  />
</p>

LSTM's address this problem with the following innovations:

### Gates and Cells

**Cells** are responsible for memory retention over irregular and long periods of time. Information about previous observations are stored in the cell state. The network can add or remove information from the cell state through devices that permit selective input or modification of the cell state.

Mathematically, the cell state at a time step **t** is a function of the cell state of the previous time step **t-1** as well as a product involving the *input gate* and a TanH activation of the input, previous hidden state and a bias.

Gates serving various purposes play an important role in LSTM models. These gates are **forget gates**, **input gates**, and **output gates** and can be thought of as *regulators* of the flow of information through the LSTM.

**Forget gate**: The first decision made by the module is determine what will be discarded from the cell state. The sigmoid operation, indicated by the *sigma* symbol, outputs a value between 1 and 0, with a 0 indicating "forget all" and a 1 indicating "keep all".

**Input gate**: The LSTM then needs to determine what new information will be stored in the cell state.  The first component of this cell state update involves the input gate. A sigmoid activation produces this gate's output, also a function of the current inputs and the previous hidden state, and a bias.

The second part of the cell state update involves a TanH activation of the previous time step's hidden state and current state's inputs, the output of which is then combined with the output of the input gate.

**Output gate**: The final gated flow occurs through the output gate. The output is a filtered version of the cell state. It is another sigmoid activation of the inputs of the current time step and the hidden state of the previous time step.

Finally, the new hidden state is composed of a TanH activation of the new cell state, and the output from the output gate.

