from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# Data paths
X_TRAIN_PATH = '../data/data_UNet/images'
Y_TRAIN_PATH = '../data/data_UNet/masks'
TRAIN_PLOTS_PATH = '../train/plots'
TRAIN_MODELS_PATH = '../train/models'


# ROI detection
NUM_CLASSES_MASK = 2
BACKBONE = 'resnet50'
OPTIMIZER = Adam()
LEARNING_RATE = 0.0001
LR_SCHEDULE = ExponentialDecay(LEARNING_RATE,
                               decay_steps=100000,
                               decay_rate=0.8,
                               staircase=True)
EARLYSTOPPING = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              verbose=1,
                              mode='auto',
                              baseline=None,
                              restore_best_weights=True)
BATCHSIZE = 8
EPOCHS = 50

# ROI extraction
IMAGES_CSV_PATH = '../data/data.csv'
IMAGES_TO_EXTRACT_ROI_PATH = '../data/OAI_dataset/extracted_images'
ROI_SAVE_PATH = '../data'
ROI_VIZ_SAVE_PATH = '../data_UNet/roi_viz'
ROI_MODEL_PATH = '../train/models/KOA_roi_resnet50.hdf5'

# Classification config
# IMAGES_DIR = '../data/'
# CALLBACK = EarlyStopping(monitor='val_loss',
#                          min_delta=0,
#                          patience=5,
#                          verbose=1,
#                          mode='auto',
#                          baseline=None,
#                          restore_best_weights=True)


