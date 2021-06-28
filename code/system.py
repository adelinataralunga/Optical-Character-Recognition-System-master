"""Optical Character Recognition System using
LDA for dimensionality reduction and 
k-nearest neighbours for classification

"""
import numpy as np
import utils.utils as utils
import scipy.linalg
from scipy import stats
import string
import cv2


def reduce_dimensions(feature_vectors_full, model):
    """Returns best 10 features using LDA

    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    """
    if (not('vector' in model)):
        labels_train = np.array(model['labels_train'])
        characters = np.unique(labels_train)
        N = characters.shape[0]

        # compute mean vectors for each class
        mean_vectors = []
        for c in np.nditer(characters):
            mean_vectors.append(np.mean(feature_vectors_full[labels_train == c], axis=0))

        # scatter within classes matrix
        sw = np.zeros((feature_vectors_full.shape[1], feature_vectors_full.shape[1]))
        for cl, mv in zip(characters, mean_vectors):
            class_sc_mat = np.zeros((feature_vectors_full.shape[1], feature_vectors_full.shape[1]))
            for row in feature_vectors_full[labels_train == cl]:
                row = row.reshape(feature_vectors_full.shape[1], 1)
                mv = mv.reshape(feature_vectors_full.shape[1], 1)
                class_sc_mat += (row-mv).dot((row-mv).T)
            sw += class_sc_mat

        overall_mean = np.mean(feature_vectors_full, axis=0)
        # compute scatter between classes matrix
        sb = np.zeros((feature_vectors_full.shape[1], feature_vectors_full.shape[1]))
        for i, mv in enumerate(mean_vectors):
            n = feature_vectors_full[labels_train == characters[i], :].shape[0]
            mv = mv.reshape(feature_vectors_full.shape[1], 1)
            overall_mean = overall_mean.reshape(feature_vectors_full.shape[1], 1)
            sb += n * (mv - overall_mean).dot((mv - overall_mean).T)

        m = np.linalg.inv(sw).dot(sb)
        N = m.shape[0]
        w, v = scipy.linalg.eigh(m, eigvals=(N - 11, N - 2))
        v = np.fliplr(v)
        model['vector'] = v.tolist()

    vector = model['vector']
    ldatrain_data = np.dot(feature_vectors_full, vector)

    return ldatrain_data


"""def reduce_dimensions(feature_vectors_full, model):
    Returns best 10 features using PCA

    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    
    if (not('vector' in model)):
        covx = np.cov(feature_vectors_full, rowvar=0)
        N = covx.shape[0]
        w, v = scipy.linalg.eigh(covx, eigvals=(N - 10, N - 1))
        v = np.fliplr(v)
        model['vector'] = v.tolist()

    vector = model['vector']
    pcatrain_data = np.dot(feature_vectors_full, vector)

    return pcatrain_data"""


def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    return height, width


def images_to_feature_vectors(images, bbox_size=None):
    """Reformat characters into feature vectors.

    Takes a list of images stored as 2D-arrays and returns
    a matrix in which each row is a fixed length feature vector
    corresponding to the image.abs

    Params:
    images - a list of images stored as arrays
    bbox_size - an optional fixed bounding box size for each image
    """

    # If no bounding box size is supplied then compute a suitable
    # bounding box by examining sizes of the supplied images.
    if bbox_size is None:
        bbox_size = get_bounding_box_size(images)

    bbox_h, bbox_w = bbox_size
    nfeatures = bbox_h * bbox_w
    fvectors = np.empty((len(images), nfeatures))
    for i, image in enumerate(images):
        padded_image = np.ones(bbox_size) * 255
        h, w = image.shape
        h = min(h, bbox_h)
        w = min(w, bbox_w)
        padded_image[0:h, 0:w] = image[0:h, 0:w]
        fvectors[i, :] = padded_image.reshape(1, nfeatures)
    return fvectors


# The three functions below this point are called by train.py
# and evaluate.py and need to be provided.

def process_training_data(train_page_names):
    """Perform the training stage and return results in a dictionary.

    Params:
    train_page_names - list of training page names
    """
    print('Reading data')
    images_train = []
    labels_train = []
    for page_name in train_page_names:
        images_train = utils.load_char_images(page_name, images_train)
        labels_train = utils.load_labels(page_name, labels_train)
    labels_train = np.array(labels_train)

    print('Extracting features from training data')
    bbox_size = get_bounding_box_size(images_train)
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)
    # add noise to training data
    fvectors_train_full = np.random.normal(fvectors_train_full, 50)

    with open('english-words.txt') as f:
        dictionary = [word.rstrip() for word in f]

    model_data = dict()
    model_data['dict'] = dictionary
    model_data['labels_train'] = labels_train.tolist()
    model_data['bbox_size'] = bbox_size

    print('Reducing to 10 dimensions')
    fvectors_train = reduce_dimensions(fvectors_train_full, model_data)

    model_data['fvectors_train'] = fvectors_train.tolist()
    return model_data


def load_test_page(page_name, model):
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    bbox_size = model['bbox_size']
    images_test = utils.load_char_images(page_name)

    # remove noise from data
    for n, image in enumerate(images_test):
        if(image.shape[0] != 0 and image.shape[1] != 0):
            images_test[n] = cv2.medianBlur(image, 3)

    fvectors_test = images_to_feature_vectors(images_test, bbox_size)
    # Perform the dimensionality reduction.
    fvectors_test_reduced = reduce_dimensions(fvectors_test, model)
    return fvectors_test_reduced


"""def classify_page(page, model):
    Classifier using nearest neighbour

    parameters:

    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    
    train = np.array(model['fvectors_train'])
    labels_train = np.array(model['labels_train'])
    # nearest neighbour
    x= np.dot(page, train.transpose())
    modtest=np.sqrt(np.sum(page * page, axis=1))
    modtrain=np.sqrt(np.sum(train * train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose()) # cosine distance
    nearest=np.argmax(dist, axis=1)
    label = labels_train[nearest]
    return label"""


def classify_page(page, model):
    """Classifier using k-nearest neighbour

    parameters:

    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """
    train = np.array(model['fvectors_train'])
    labels_train = np.array(model['labels_train'])

    x = np.dot(page, train.transpose())
    modtest = np.sqrt(np.sum(page * page, axis=1))
    modtrain = np.sqrt(np.sum(train * train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose())  # cosine distance
    # get indices with minimum distances
    nearest = np.sort(np.argpartition(dist, -9, axis=1)[:, -9:], axis=1)
    d = np.take(labels_train, nearest)  # replace with labels
    arr, _ = stats.mode(d, axis=1)  # for each row get most frequent label
    arr = arr.reshape((page.shape[0], ))
    return arr


def remove_punctuation(word):
    """Removes punctuation from a word

    parameters:

    word - the word from which the punctuation needs to be removed
    """
    word = "".join(l for l in word if l not in string.punctuation)

    return word


def add_punctuation(s1, s2):
    """Adds punctuation in the second word in the positions from the first one

    parameters:

    s1 - the word from which to take the punctuation and positions
    s2 - the word in which to insert punctuation
    """
    s2 = list(s2)
    for i, l in enumerate(s1):
        if l in string.punctuation:
            s2.insert(i, l)

    return ''.join(s2)


def closest_match(word, dictionary):
    """Returns the closest match for a word in the given dictionary

    parameters:

    word - string, the word to be replaced
    dictionary - list of strings, the dictionary
    in which to find the closest match
    """
    same_length = filter(lambda x: len(x) == len(word), dictionary)
    # a maximum of 1 character must be changed in very short words
    for x, match in enumerate(same_length):
        if (hamming(match, word) == 1):
            return match
    # maximum 2 characters must be changed in words of length 1-3
    for x, match in enumerate(same_length):
        if (len(match) < 4 and hamming(match, word) <= 2):
            return match
    # a maximum of 3 characters changed is allowed
    for x, match in enumerate(same_length):
        if (len(match) < 8 and hamming(match, word) <= 3):
            return match

    return word


def hamming(s1, s2):
    """Calculate the Hamming distance between two strings

    parameters:

    s1 - first string
    s2 - second string
    """
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def correct_word(word, dictionary):
    """Corrects the errors in a word

    parameters:

    word - string, the word to be corrected
    dictionary - list of strings, the dictionary in the word is searched for
    """
    match = word

    # check if the word is not in the dictionary
    if(not(any(remove_punctuation(word).lower() == w for w in dictionary))):
        match = closest_match(remove_punctuation(word).lower(), dictionary)
        match = add_punctuation(word, match)
        if(word.istitle()):
            match = match.capitalize()
    return match


def correct_errors(page, labels, bboxes, model):
    """Corrects the misclassified labels

    parameters:

    page - 2d array, each row is a feature vetcor to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array,each row gives the 4 bouding boox coords of the character
    model - dictionary, stores the output of the training stage
    """
    dictionary = model['dict']

    widths = bboxes[:, 2]-bboxes[:, 0]  # get boxes widths
    diffs = np.diff(bboxes[:, 0])
    spaces = diffs[:] - widths[:(diffs.size)]  # spaces between boxes
    # get positions at which each word ends
    ends = np.where((spaces > 6) | (spaces < -50))

    start = 0
    for i in np.nditer(ends):
        word = labels[start:(i + 1)]  # get each word as an array of characters
        st = ''.join(word)
        cw = correct_word(st, dictionary)
        cw = np.array(list(cw))  # separate the corrected word into characters
        labels[start:(i + 1)] = cw  # replace the labels
        start = i + 1

    return labels
