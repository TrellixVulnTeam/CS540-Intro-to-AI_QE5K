import tensorflow as tf
from tensorflow import keras


def get_dataset(training=True):
    """
    :param training: an optional boolean argument (default value is True for training dataset)
    :return:two NumPy arrays for the train_images and train_labels
    """
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    if training:
        return train_images, train_labels
    else:
        return test_images, test_labels


def print_stats(train_images, train_labels):
    """ This function will print several statistics about the data
    :param train_images: dataset
    :param train_labels: labels
    :return: None
    """
    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
    print(len(train_images))
    print(str(len(train_images[0])), str(len(train_images[0][0])), sep="x")
    dic = {}
    for label in train_labels:
        if label not in dic:
            dic[label] = 0
        dic[label] += 1
    for i in range(len(dic)):
        print(str(i) + ". " + class_names[i] + " - " + str(dic[i]))


def build_model():
    """ takes no arguments and returns an untrained neural network model
    :return: an untrained neural network model
    """
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(10))
    opt = keras.optimizers.SGD(learning_rate=0.001)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    return model


def train_model(model, train_images, train_labels, T):
    """ takes the model produced by the previous function and the dataset and
    labels produced by the first function and trains the data for T epochs
    :param model: the model produced by the previous function
    :param train_images: dataset
    :param train_labels: labels
    :param T: epochs
    :return: None
    """
    model.fit(train_images, train_labels, epochs=T)


def evaluate_model(model, test_images, test_labels, show_loss=True):
    """ takes the trained model produced by the previous function and the test
    image/labels, and prints the evaluation statistics as described below (displaying
    the loss metric value if and only if the optional parameter has not been set to False);
    :param model: the trained model produced by the previous function
    :param test_images: test image
    :param test_labels: test labels
    :param show_loss: metric
    :return: None
    """
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    if show_loss:
        print("Loss: {:.4f}".format(round(test_loss, 4)))
    print("Accuracy: {:.2f}%".format(round(100 * test_accuracy, 2)))


def predict_label(model, test_images, index):
    """ takes the trained model and test images, and prints the top 3 most likely labels
    for the image at the given index, along with their probabilities;
    :param model: the trained model
    :param test_images: test images
    :param index: index
    :return: None
    """
    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
    extracted = model.predict(test_images)[index]
    dic = {}
    for i in range(len(class_names)):
        dic[class_names[i]] = extracted[i]
    top_three = sorted(class_names, key=lambda x: dic[x], reverse=True)[:3]
    for top in top_three:
        print(top + ": {:.2f}%".format(round(100 * dic[top], 2)))
