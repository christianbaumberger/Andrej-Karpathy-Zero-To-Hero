## Makemore Learnings

- Make more of Things
- Make new names
- character level language model
- it knows how to predict the next character
- Bigrams are models which gives you the next character/token/word in a sequence given the last character/token/word
- Nice way in python to run over a string with always 2 characters -> zip(word, word[1:])
- Simple character Bigram is counting the number of how often the bigrams occur, can be done with a dict, but is more conventient with a 2-dim way -> torch.tensor
- Maximum likelihood estimation is the product of the individual propabilities
- A product of probabilities gives a too small number
- The log transforms the probabilites into numbers between -inf and 0
- Through the logarithm, the product turns into a sum of logprobs
- Since we want to minimize, we take the negative log likelihood
- This is our loss NLL
- logits are the unnormalized outputs of a neural network which is made for classification
- logits can be interpreted as log counts -> bigram with count matrix
- apply the exp function to the logits to get something similar to counts and normalize it to get probabilities
- This is called softmax
- With L2 regularization we can smooth the model -> add squared weights of NN to loss
- What is one hot encoding? -> just the selection of a row of the W matrix

### Torch

Normalize each row of a matrix to sum to one:

- P = torch.randn((10,10)))
- P /= P.sum(dim=1, keepdim=True)

### Lecture 3 Barch Normalization

- how to scale a weight matrix so that with a matrix multiplication it preserves a zero mean standard gaussian distr
- divide by the sqrt of fan-in -> w / 10**0.5 -> here fan-in is 10 (number of input elements)
- modern innovation like residual connections and normaliations (batch/layer, ...) make it less important to exactly initilize the layers
- we dont want our activations to be too squashed or to spread, because of the nonlinearities
- we want roughly gaussia activations throughout the neural net
- so we can scale the weight matrices in order to control the statistics of the activations
- but this careful fideling with the inits of the weights is not handable through a large deep net
- Therefore the concept of batch normalization can be used
- batch normalization: just center your data and make unit variance -> in addition add a gain and a bias term
- this is all possible, since all those operations are differentiable
- one disadvantage: all the input samples in the batch gets coupled -> not a wanted feature
- to do the inference, we need to keep track of the mean and std during training
- there are better alternatives than batch normalization like group and layer normalization
- why is the gain of 5/3 necessary (on the torchifying examples) on the linear layers:
  - if it would be one, it would be ok for the first layer, but since there are tanh nonlinearities inbetween the linear layers, it squashes the activations, therefore some gain is needed in order to avoid the squashing
- so we neeed first the kaiming initialization on the linear weight matrices and then also need a gain on the linear layers and also a multiply with 0.1 on the last layer for the softmax
- we also look at the gradient distribution and the important thing here is, that the gradient distribution over the different layers should be the same, not shrinking or exploding
- if the gain is too small or too large, the gradient distributions are messed up
- with the batch norm layer in place, it is much more robust to the gain and weights initialization, maybe we have to change the learning rate
