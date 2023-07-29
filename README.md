# Residual Networks (ResNets) - Readme

## Introduction

In this project we will be building a very deep convolutional network using Residual Networks (ResNets). The ResNets architecture, introduced by [He et al.](https://arxiv.org/pdf/1512.03385.pdf), allows us to train much deeper neural networks than were previously feasible. Very deep networks have the potential to represent very complex functions and learn features at multiple levels of abstraction. However, training such deep networks can be challenging due to vanishing gradients, which can slow down gradient descent during backpropagation.

The key innovation of ResNets is the introduction of "shortcut" or "skip connections." These connections allow the model to skip over certain layers and directly connect earlier layers to later layers, facilitating the flow of gradients and making it easier to learn an identity function. As a result, ResNets can be stacked to form very deep networks with little risk of harming training set performance.

By the end of this project, we will be able to:
1. Implement the basic building blocks of ResNets in a deep neural network using Keras.
2. Put together these building blocks to implement and train a state-of-the-art neural network for image classification.
3. Implement a skip connection in your network.

Let's dive into the details of implementing the ResNet building blocks and building a ResNet-50 model.

## Building a Residual Network

### 1. The Identity Block

The identity block is the standard building block used in ResNets, where the input activation has the same dimensions as the output activation. This block skips over 3 hidden layers. The individual steps of the identity block are as follows:

#### First component of the main path:
- Conv2D layer with 𝐹1 filters of shape (1,1) and a stride of (1,1), using "valid" padding and random uniform initialization.
- BatchNormalization to normalize the 'channels' axis.
- ReLU activation function.

#### Second component of the main path:
- Conv2D layer with 𝐹2 filters of shape (𝑓,𝑓) and a stride of (1,1), using "same" padding and random uniform initialization.
- BatchNormalization to normalize the 'channels' axis.
- ReLU activation function.

#### Third component of the main path:
- Conv2D layer with 𝐹3 filters of shape (1,1) and a stride of (1,1), using "valid" padding and random uniform initialization.
- BatchNormalization to normalize the 'channels' axis.

#### Final step:
- Add the output from the third component (X) to the input activation (X_shortcut).
- Apply the ReLU activation function.

### 2. The Convolutional Block

The convolutional block is used when the input and output dimensions don't match up. It includes a Conv2D layer in the shortcut path to resize the input so that the dimensions match up in the final addition step. The details of the convolutional block are as follows:

#### First component of the main path:
- Conv2D layer with 𝐹1 filters of shape (1,1) and a stride of (s,s), using "valid" padding and glorot_uniform initialization.
- BatchNormalization to normalize the 'channels' axis.
- ReLU activation function.

#### Second component of the main path:
- Conv2D layer with 𝐹2 filters of shape (f,f) and a stride of (1,1), using "same" padding and glorot_uniform initialization.
- BatchNormalization to normalize the 'channels' axis.
- ReLU activation function.

#### Third component of the main path:
- Conv2D layer with 𝐹3 filters of shape (1,1) and a stride of (1,1), using "valid" padding and glorot_uniform initialization.
- BatchNormalization to normalize the 'channels' axis.

#### Shortcut path:
- Conv2D layer with 𝐹3 filters of shape (1,1) and a stride of (s,s), using "valid" padding and glorot_uniform initialization.
- BatchNormalization to normalize the 'channels' axis.

#### Final step:
- Add the output from the shortcut path to the main path.
- Apply the ReLU activation function.

### 3. ResNet-50 Model

The ResNet-50 model consists of several stages and blocks. The architecture is as follows:

#### Stage 0:
- Zero-padding pads the input with a pad of (3,3).

#### Stage 1:
- Conv2D layer with 64 filters of shape (7,7) and a stride of (2,2), using "valid" padding.
- BatchNormalization to normalize the 'channels' axis.
- MaxPooling layer with a (3,3) window and a (2,2) stride.

#### Stage 2:
- Three convolutional blocks using three sets of filters of size [64,64,256], with "f" as 3 and "s" as 1.
- Two identity blocks using three sets of filters of size [64,64,256], with "f" as 3.

#### Stage 3:
- Three convolutional blocks using three sets of filters of size [128,128,512], with "f" as 3 and "s" as 2.
- Three identity blocks using three sets of filters of size [128,128,512], with "f" as 3.

#### Stage 4:
- Three convolutional blocks using three sets of filters of size [256, 256, 1024], with "f" as 3 and "s" as 2.
- Five identity blocks using three sets of filters of size [256, 256, 1024], with "f" as 3.

#### Stage 5:
- Three convolutional blocks using three sets of filters of size [512, 512, 2048], with "f" as 3 and "s" as 2.
- Two identity blocks using three sets of filters of size [512, 512, 2048], with "f" as 3.

#### Final steps:
- 2D Average Pooling with a window of shape (2,2).
- Flatten layer to prepare for the fully connected (Dense) layer.
- Fully connected layer with a softmax activation to reduce the input to the number of classes.

### Implementation

In the implementation, we will be using Keras to build the ResNet-50 model and its building blocks. we'll implement both the identity block and the convolutional block and then combine them to create the complete ResNet-50 model. Each building block will consist of Conv2D, BatchNormalization, ReLU activation, and Add operations.

Let's get started and build a powerful ResNet-50 model for image classification using Keras!
