import preprocess
import _pickle as pkl
import numpy as np
import os


TRAIN_PATH = ""


def create_and_save(data_dir, labels_dir, saved_file_path):
    images = []
    labels = []
    landmarks = []
    max_label = []
    path_to_images = os.path.join(TRAIN_PATH, data_dir)
    path_to_labels = os.path.join(TRAIN_PATH, labels_dir)
    with open(os.path.join(path_to_labels, "angles.csv"), "r") as f:
        mylabels = f.readlines()
    with open(os.path.join(path_to_labels, "landmarks.csv"), "r") as f:
        mylandmarks = f.readlines()
    with open(os.path.join(path_to_labels, "filenames.csv"), "r") as f:
        filenames = f.readlines()
    paired_labels = dict(zip(filenames, mylabels))
    paired_landmarks = dict(zip(filenames, mylandmarks))
    for image_file in paired_labels.keys():
        label = paired_labels[image_file]
        landmark = paired_landmarks[image_file]
        coords = label.split(",")
        coords = [float(x) for x in coords]
        max_label.append(max(coords))
        lm = landmark.split(",")
        lm = np.array([float(x) for x in lm], dtype=np.float32)
        labels.append(coords)
        landmarks.append(lm)
        image = preprocess.preprocess(path_to_images, image_file.strip("\n"))
        images.append(image)
    labels = np.array(labels, dtype=np.float32)
    images = np.array(images, dtype=np.float32)
    landmarks = np.array(landmarks, dtype=np.float32)
    max_label = np.array(max_label)
    
    with open(os.path.join(TRAIN_PATH, saved_file_path), 'wb') as f:
        pkl.dump([images, landmarks, labels, max_label], f)
        
        
    
create_and_save('data/training', 'labels/training', 'train.pkl')
create_and_save('data/test', 'labels/test', 'test.pkl')
