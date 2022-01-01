from configs import config_cls
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sn

TRAIN_MODELS_PATH = str(config_cls.TRAIN_MODELS_PATH)+'/KOA_cls_densenet121_2steps_aug_weights_focal_kappa_0.8724_best.hdf5'

# Create image data generator
datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                             validation_split=0.2)

train_generator = datagen.flow_from_directory(directory=config_cls.IMAGES_DIR,  # images directory
                                              target_size=(224, 224),
                                              batch_size=16,
                                              seed=42,
                                              subset='training')  # set as training data
valid_generator = datagen.flow_from_directory(directory=config_cls.IMAGES_DIR,  # same directory as training data
                                              # directory=config.IMAGES_DIR,  # same directory as training data
                                              target_size=(224, 224),
                                              seed=42,
                                              batch_size=16,
                                              subset='validation',  # set as validation data
                                              shuffle=False)

train_steps = train_generator.samples // train_generator.batch_size
validation_steps = valid_generator.samples // valid_generator.batch_size

model = load_model(TRAIN_MODELS_PATH, compile=False)

# Confusion Matrix
y_valid = []
y_pred = []

# Retrieve true labels from validation generator and get prediction from trained model
for step in range(validation_steps):
    X, y = valid_generator[step]
    prediction = model.predict(X)

    for i in range(y.shape[0]):
        y_valid.append(np.argmax(y[i]))
        y_pred.append(np.argmax(prediction[i]))

y_valid = np.asarray(y_valid)
y_pred = np.asarray(y_pred)

confusion = confusion_matrix(y_valid, y_pred)
cfm = pd.DataFrame(confusion, index=range(5), columns=range(5))

plt.figure(figsize=(10, 7))
sn.heatmap(cfm, annot=True)
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('EfficientNet-B4 - Confusion Matrix')
plt.savefig(f'{config_cls.BAYES_OPT_RESULTS}/KOA_cls_densenet121_2steps_aug_weights_focal_confmatrix_plot.png')
plt.show()
