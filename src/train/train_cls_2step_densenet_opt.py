from configs import config_cls
from utils.util import *
from matplotlib import pyplot as plt
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout
from keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
import tensorflow_addons as tfa
from sklearn.metrics import confusion_matrix
from bayes_opt import BayesianOptimization
import numpy as np
import pandas as pd
import seaborn as sn


# Create image data generator
datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                             validation_split=0.2)

train_generator = datagen.flow_from_directory(directory=config_cls.IMAGES_DIR,  # images directory
                                              target_size=(224, 224),
                                              seed=42,
                                              batch_size=16,
                                              subset='training')  # set as training data
valid_generator = datagen.flow_from_directory(directory=config_cls.IMAGES_DIR,  # same directory as training data
                                              target_size=(224, 224),
                                              seed=42,
                                              batch_size=16,
                                              subset='validation',  # set as validation data
                                              shuffle=False)

# Obtain class weights
class_weights = dict(enumerate(class_weight.compute_class_weight(
    'balanced',
    np.unique(train_generator.classes),
    train_generator.classes)
)
)

# Define metrics
cohen_kappa = tfa.metrics.CohenKappa(
    num_classes=5,
    name='cohen_kappa',
    weightage='quadratic'
)


train_steps = train_generator.samples // train_generator.batch_size
validation_steps = valid_generator.samples // valid_generator.batch_size

# Define metrics
f1 = tfa.metrics.F1Score(5)


# Define function to unfreeze model layers and compile
def unfreeze_model(model, learning_rate=0.0001):
    model.trainable = True

    # Unfreeze Convolutional Layers
    for layer in model.layers:
        if 'conv' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  # loss='categorical_crossentropy',
                  loss=categorical_focal_loss(list(class_weights.values()), gamma=2),
                  # loss=categorical_focal_loss(alpha=[[.25, .25, .25, .25, .25]], gamma=2),
                  # loss=categorical_focal_loss(alpha=[[.0478, .1045, .0724, .1445, .6311]], gamma=2),
                  metrics=[['acc', f1, cohen_kappa]])
    # metrics=['acc'])
    print('Conv layers in model set to trainable.\n')


##########################################################################################
# OPTIMIZATION
def cnn_opt(neurons, learning_rate, dropout):
    optimizer = config_cls.OPTIMIZER
    neurons = round(neurons)
    dropout = round(dropout, 2)

    # Create the model
    def create_cnn(model_arch=DenseNet121):
        nn = model_arch(include_top=False,
                        input_shape=(224, 224, 3),
                        pooling='avg')
        nn.trainable = False
        x = Dense(neurons, activation='relu')(nn.output)
        x = Dropout(rate=dropout)(x)
        x = Dense(5, activation='softmax')(x)

        model = Model(inputs=nn.input, outputs=x)
        model.compile(optimizer=optimizer,
                      # loss='categorical_crossentropy',
                      loss=categorical_focal_loss(list(class_weights.values()), gamma=2),
                      metrics=[['acc', f1, cohen_kappa]])
        return model

    # STEP 1: train model for 3 epochs with backbone weights frozen
    pretrain_epochs = config_cls.PRETRAIN_EPOCHS
    epochs = config_cls.EPOCHS
    train_steps = train_generator.samples // train_generator.batch_size
    validation_steps = valid_generator.samples // valid_generator.batch_size

    model = create_cnn(DenseNet121)
    history_1 = model.fit(x=train_generator,
                          steps_per_epoch=train_steps,
                          validation_data=valid_generator,
                          validation_steps=validation_steps,
                          epochs=pretrain_epochs,
                          class_weight=class_weights
                          )

    # STEP 2: unfreeze conv layers weights and train until epoch limit or early stop.
    lr = round(learning_rate, 5)
    unfreeze_model(model, learning_rate=lr)
    CALLBACK_LR = LossLearningRateScheduler(base_lr=lr, lookback_epochs=2)
    history_2 = model.fit(x=train_generator,
                          steps_per_epoch=train_steps,
                          validation_data=valid_generator,
                          validation_steps=validation_steps,
                          epochs=epochs,
                          initial_epoch=pretrain_epochs,
                          callbacks=[config_cls.CALLBACK_ES, CALLBACK_LR],
                          class_weight=class_weights
                          )
    # Record stopping epoch, valid loss and metrics
    early_stop_epochs = len(history_2.history['val_loss']) - config_cls.PATIENCE - 1
    list_early_stop_epochs.append(pretrain_epochs + early_stop_epochs)
    validation_loss = history_2.history['val_loss'][early_stop_epochs]
    list_validation_loss.append(validation_loss)
    list_validation_acc.append(history_2.history['val_acc'][early_stop_epochs])
    list_validation_f1_score.append(history_2.history['val_f1_score'][early_stop_epochs])
    list_validation_cohen_kappa_score.append(history_2.history['val_cohen_kappa'][early_stop_epochs])

    # Update best loss and metrics
    best_valid_cohen_kappa_score = max(list_validation_cohen_kappa_score)
    best_idx = np.argmax(list_validation_cohen_kappa_score)
    best_valid_loss = list_validation_loss[best_idx]

    # Save best model
    if history_2.history['val_cohen_kappa'][early_stop_epochs] >= best_valid_cohen_kappa_score and \
            history_2.history['val_loss'][early_stop_epochs] <= best_valid_loss:
        model.save(
            f'{config_cls.BAYES_OPT_RESULTS}/KOA_cls_densenet121_2steps_aug_weights_focal_kappa_{round(best_valid_cohen_kappa_score, 4)}_best.hdf5')
        print('\nBest model has been saved.\n')

        # Plot training/validation loss/accuracy
        loss = history_1.history['loss'] + history_2.history['loss']
        val_loss = history_1.history['val_loss'] + history_2.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title(f'KOA Classification - Categorical Focal Loss (DenseNet121)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{config_cls.BAYES_OPT_RESULTS}/KOA_cls_densenet121_2steps_aug_weights_focal_loss_plot.png')
        plt.show()

        kappa_score = history_1.history['cohen_kappa'] + history_2.history['cohen_kappa']
        val_kappa_score = history_1.history['val_cohen_kappa'] + history_2.history['val_cohen_kappa']
        epochs = range(1, len(kappa_score) + 1)
        plt.plot(epochs, kappa_score, 'b', label='Training Cohen Kappa score')
        plt.plot(epochs, val_kappa_score, 'r', label='Validation Cohen Kappa score')
        plt.title(f'KOA Classification - Cohen Kappa score (DenseNet121)')
        plt.xlabel('Epochs')
        plt.ylabel('Cohen Kappa score')
        plt.legend()
        plt.savefig(f'{config_cls.BAYES_OPT_RESULTS}/KOA_cls_densenet121_2steps_aug_weights_focal_kappa_plot.png')
        plt.show()

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
        plt.title('DenseNet121 - Confusion Matrix')
        plt.savefig(f'{config_cls.BAYES_OPT_RESULTS}/KOA_cls_densenet121_2steps_aug_weights_focal_confmatrix_plot.png')
        plt.show()

    # Bayes optimization is a maximization algorithm. In order to minimize
    # validation_loss, return (1 - validation_loss)
    bayes_opt_score = 1.0 - validation_loss

    return bayes_opt_score

##########################################################################################

# Initiate placeholders for results recording
list_early_stop_epochs = []
list_validation_loss = []
list_validation_acc = []
list_validation_f1_score = []
list_validation_cohen_kappa_score = []

# Set of hyperparameters
params_nn = config_cls.PARAMS_NN

# Run Bayesian Optimization
nn_bayes_opt = BayesianOptimization(f=cnn_opt,
                                    pbounds=params_nn,
                                    random_state=42)
nn_bayes_opt.maximize(init_points=10, n_iter=1)

print('\nBest result:', nn_bayes_opt.max)

# Collection of Bayesian Optimization results
list_dfs = []
counter = 0
for result in nn_bayes_opt.res:
    df_temp = pd.DataFrame.from_dict(data=result['params'],
                                     orient='index',
                                     columns=['trial' + str(counter)]).T
    df_temp['bayes opt error'] = result['target']

    df_temp['epochs'] = list_early_stop_epochs[counter]
    df_temp['validation_loss'] = list_validation_loss[counter]
    df_temp['validation_acc'] = list_validation_acc[counter]
    df_temp['validation_cohen_kappa'] = list_validation_cohen_kappa_score[counter]

    list_dfs.append(df_temp)

    counter += 1

df_results = pd.concat(list_dfs, axis=0)
df_results.to_csv(config_cls.BAYES_OPT_RESULTS + 'df_densenet121_aug_focal_bayes_opt_results_parameters.csv')
