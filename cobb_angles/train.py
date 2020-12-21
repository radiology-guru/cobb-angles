import train_spine_model as M


# you will need to customize PATH_TO_IMAGES to where you have uncompressed
# NIH images
PATH_TO_IMAGES = ""
PATH_TO_MODEL = ""

WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.01
preds, aucs = M.train_cnn(PATH_TO_IMAGES, PATH_TO_MODEL, LEARNING_RATE, WEIGHT_DECAY)

