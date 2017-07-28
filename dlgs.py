def create_tf_checkpoint(sess, path, version = 1):
    """
    Creates a tensoflow checkpoint 
    : sess: The TensorFlow session you want to save
    : path: The name of the file to save the session to
    : version: The version of the checkpoint, by default is 1, so you can create different checkpoints in time for your model    
    """  
    saver = tf.train.Saver()
    save_path = saver.save(sess, path, global_stet = version)

def restore_tf_checkpoint(sess, path):
    loader = tf.train.import_meta_graph(path + '.meta')
    loader.restore(sess, path)

def get_tf_weights(shape, stddev = 0.1):
    import tensorflow as tf
    weights = tf.Variable(tf.truncated_normal(shape=shape,stddev = stddev))
    return weights

def get_tf_biases(num_outputs):
    biases = tf.Variable(tf.zeros(num_outputs))
    return biases


def get_bag_of_words(text):
    from collections import Counter
    return Counter(text.split())

def normalize_image(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data. 
    : return: Numpy array of normalize data
    """
    normalized_x_min = 0.1
    normalized_x_max = 0.9
    grayscale_min = 0
    grayscale_max = 255
    
    normalized_x = normalized_x_min + ( 
        ( (x - grayscale_min)*
        (normalized_x_max - normalized_x_min) )/
        ( grayscale_max - grayscale_min ))       
    
    return normalized_x

def one_hot_encode_sk(x):
    """   
    : x: List of sample Labels
    : return: array of one-hot encoded labels
    """        
    from sklearn import preprocessing
    
    labels_vecs = preprocessing.LabelBinarizer()
    labels_vecs.fit_transform(labels)
    return labels_vecs
    
def one_hot_encoder_np(x):
    """   
    : x: List of sample Labels
    : return:Numpy array of one-hot encoded labels
    """
    import numpy as np
    return np.eye(number_of_labels)[x]