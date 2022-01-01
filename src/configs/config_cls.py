from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# Classification config
IMAGES_DIR = '../data/data'
TRAIN_MODELS_PATH = './train/models'
BAYES_OPT_RESULTS = './train/BayesOpt_results/'

PATIENCE = 3
CALLBACK_ES = EarlyStopping(monitor='val_loss',
                            min_delta=0,
                            patience=PATIENCE,
                            verbose=1,
                            mode='auto',
                            baseline=None,
                            restore_best_weights=True)
OPTIMIZER = Adam(learning_rate=0.001)
PRETRAIN_EPOCHS = 3
EPOCHS = 50
PARAMS_NN = {
    'neurons': (1024, 2048),
    'learning_rate': (0.0001, 0.001),
    'dropout': (0.2, 0.5)
}