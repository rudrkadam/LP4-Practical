# LP4 Assignments for BE(IT) Semester 7 of SPPU

### Assignment 1: Study of Deep Learning Packages (TensorFlow, Keras, Theano, PyTorch)

#### Purpose
This assignment is about understanding the different deep learning frameworks. These frameworks help build, train, and deploy deep learning models efficiently.

#### Key Points
1. **TensorFlow**: A comprehensive library by Google that’s widely used for deep learning tasks. It supports both low-level operations (for customization) and high-level APIs (like Keras) for ease of use.
2. **Keras**: Initially an independent library, it’s now integrated into TensorFlow. Keras is highly user-friendly, providing a high-level API to design, build, and train deep learning models with less code.
3. **Theano**: One of the original deep learning libraries, Theano is designed for numerical computation. It allows defining, optimizing, and evaluating mathematical expressions but is now mostly used as a backend for other libraries like Keras.
4. **PyTorch**: Developed by Facebook, PyTorch is popular for its flexibility, especially in research. It has a dynamic computation graph (meaning you can change the network structure on the fly), making it ideal for experimentation.

#### Tasks
- Install these packages and explore their syntax, major features, and usage.
- Document their functionalities and unique aspects, such as how TensorFlow focuses on deployment, while PyTorch emphasizes flexibility.

---

### Assignment 2: Implementing Feedforward Neural Networks

#### Purpose
To build a simple Feedforward Neural Network (also known as a Multilayer Perceptron) using Keras and TensorFlow for tasks like image classification.

#### Key Steps
1. **Import necessary packages**: Use TensorFlow and Keras for building and training the model.
2. **Load training and testing data**: Choose a dataset like **MNIST** (handwritten digits) or **CIFAR-10** (small color images). These datasets are preloaded in Keras, making them easy to use.
3. **Define network architecture**: Build the layers using Keras. This might involve an input layer, one or more hidden layers with activation functions (like ReLU), and an output layer (e.g., softmax for classification).
4. **Train the model**: Use **Stochastic Gradient Descent (SGD)** as the optimization method to adjust weights based on the error.
5. **Evaluate the network**: Test the model on unseen data and measure its accuracy.
6. **Plot the training loss and accuracy**: Visualize how the model improves over epochs, helping understand if it’s learning effectively or overfitting.

---

### Assignment 3: Building an Image Classification Model

#### Purpose
To create a complete image classification pipeline using deep learning techniques.

#### Key Stages
1. **Loading and Pre-processing the Image Data**: Import images and preprocess them by resizing, normalizing, and augmenting (optional) to make the data suitable for the model.
2. **Defining the Model’s Architecture**: Design the network, usually a Convolutional Neural Network (CNN) for image data. This includes layers like convolutional layers, pooling layers, and fully connected layers.
3. **Training the Model**: Use a dataset (such as CIFAR-10) to teach the model to recognize patterns in images. Define loss and optimizer (e.g., categorical cross-entropy and Adam).
4. **Estimating Model Performance**: After training, test the model on a separate test set and evaluate metrics like accuracy, precision, and recall to see how well it generalizes to new images.

---

### Assignment 4: ECG Anomaly Detection Using Autoencoders

#### Purpose
To use **Autoencoders** for detecting anomalies in ECG data, where normal data patterns are learned, and deviations are flagged as anomalies.

#### Key Steps
1. **Import Required Libraries**: Use libraries like TensorFlow/Keras and data processing tools like NumPy.
2. **Upload/Access Dataset**: Load ECG data where normal signals and anomalies are pre-labeled.
3. **Encoder-Decoder Structure**: 
   - The **Encoder** learns to compress the data to a low-dimensional “latent” representation.
   - The **Decoder** attempts to reconstruct the original data from this compressed form.
4. **Compile the Model**: Define a loss function (like mean squared error), optimizer (like Adam), and evaluation metrics to measure reconstruction accuracy.
5. **Detecting Anomalies**: When the autoencoder can’t accurately reconstruct an input (producing a high reconstruction error), this likely indicates an anomaly, as it doesn’t fit the “normal” learned pattern.

---

### Assignment 5: Implement the Continuous Bag of Words (CBOW) Model

#### Purpose
To understand how CBOW, a method used in **Natural Language Processing (NLP)**, helps predict a target word based on its surrounding words, commonly used for learning word embeddings.

#### Key Stages
1. **Data Preparation**: Start with a text corpus and preprocess it (e.g., tokenizing and creating context-target pairs). For instance, in the sentence “The cat sat on the mat,” “cat” could be the target with surrounding words as context.
2. **Generate Training Data**: Create pairs of target and context words, which CBOW will use to predict a word from its context.
3. **Train the Model**: Using a neural network, the model learns to predict a target word from given context words.
4. **Output**: Once trained, the model provides vector representations (embeddings) of words, where words with similar meanings or contexts have similar vector representations.

---

### Assignment 6: Object Detection Using Transfer Learning of CNN Architectures

#### Purpose
To apply **Transfer Learning** using a pre-trained CNN model (like ResNet, VGG) for object detection on a new, smaller dataset.

#### Key Steps
1. **Load a Pre-trained CNN Model**: Import a CNN pre-trained on a large dataset like ImageNet. These models have already learned useful features, like edges and textures, which can be adapted to a new task.
2. **Freeze Parameters**: Lock the weights of the initial layers, as they contain general feature extraction patterns.
3. **Add a Custom Classifier**: Add fully connected layers at the end of the model, which can be trained for the specific object detection task.
4. **Train Classifier Layers**: Use the dataset’s training data to fine-tune only the new classifier layers.
5. **Fine-tune Hyperparameters**: Adjust learning rate, batch size, and number of epochs. If needed, unfreeze some of the pre-trained layers for further fine-tuning, which can improve performance.
