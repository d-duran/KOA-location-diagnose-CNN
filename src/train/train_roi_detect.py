from configs import config_roi
from utils.util import read_image, read_mask, preprocess_image, preprocess_mask
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
import random
from matplotlib import pyplot as plt
import segmentation_models as sm


# Read and process training images
train_images = []
for img_path in sorted(glob.glob(os.path.join(config_roi.X_TRAIN_PATH, '*.jpg'))):
    img = read_image(img_path, aspect_ratio='square')  # (h,w,3)
    img_processed = preprocess_image(img, resize=True, size=224, equalize=True)  # (h,w,3)
    train_images.append(img_processed)
print(f'Training images imported: {len(train_images)}')

train_masks = []
for mask_path in sorted(glob.glob(os.path.join(config_roi.Y_TRAIN_PATH, '*.tiff'))):
    msk = read_mask(mask_path, aspect_ratio='square')  # (h,w)
    msk_processed = preprocess_mask(msk, resize=True, size=224)  # (h,w,1)
    train_masks.append(msk_processed)
print(f'Mask images imported: {len(train_masks)}')

# Convert lists to numpy arrays
train_images_processed = np.array(train_images)  # (n,h,w,3)
train_masks_processed = np.array(train_masks)  # (n,h,w,1)

# Data split
X_train, X_valid, y_train, y_valid = train_test_split(train_images_processed,
                                                      train_masks_processed,
                                                      test_size=0.2,
                                                      random_state=42)

##############################################################################
# Sanity check - image-mask plotting
image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12,6))
plt.subplot(121)
plt.imshow(X_train[image_number])
plt.title('X-ray processed')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (224, 224)), cmap='gray')
plt.title('Mask')
plt.savefig(f'{config_roi.TRAIN_PLOTS_PATH}/sanity_loading.png')
plt.show()
##############################################################################

# Preprocess input for backbone pre-trained model
preprocess_input = sm.get_preprocessing(config_roi.BACKBONE)
X_train = preprocess_input(X_train)
X_valid = preprocess_input(X_valid)

# Define model

# Segmentation-models library configuration settings
sm.set_framework('tf.keras')
sm.framework()

model = sm.Unet(config_roi.BACKBONE, encoder_weights='imagenet')
model.compile(optimizer=config_roi.OPTIMIZER,
              loss=sm.losses.bce_jaccard_loss,
              metrics=[sm.metrics.iou_score])

stopping = config_roi.EARLYSTOPPING

# Train model
history = model.fit(X_train,
                    y_train,
                    batch_size=config_roi.BATCHSIZE,
                    epochs=config_roi.EPOCHS,
                    verbose=1,
                    validation_data=(X_valid, y_valid),
                    callbacks=[stopping])

model.save(f'{config_roi.TRAIN_MODELS_PATH}/KOA_roi_{config_roi.BACKBONE}.hdf5')

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title(f'KOA ROI detection - BCE Jaccard Loss ({config_roi.BACKBONE})')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'{config_roi.TRAIN_MODELS_PATH}/KOA_roi_{config_roi.BACKBONE}_loss_plot.png')
plt.show()

iou_score = history.history['iou_score']
val_iou_score = history.history['val_iou_score']
epochs = range(1, len(iou_score) + 1)
plt.plot(epochs, iou_score, 'b', label='Training IOU score')
plt.plot(epochs, val_iou_score, 'r', label='Validation IOU score')
plt.title(f'KOA ROI detection - IOU score ({config_roi.BACKBONE})')
plt.xlabel('Epochs')
plt.ylabel('IOU score')
plt.legend()
plt.savefig(f'{config_roi.TRAIN_MODELS_PATH}/KOA_roi_{config_roi.BACKBONE}_iou_plot.png')
plt.show()

##############################################################################
# Sanity check - image-mask-prediction plotting
image_number = random.randint(0, len(X_valid))
plt.figure(figsize=(15,12))
plt.subplot(131)
plt.imshow(X_valid[image_number])
plt.title('X-ray processed')
plt.subplot(132)
plt.imshow(np.reshape(y_valid[image_number], (224, 224)), cmap='gray')
plt.title('Mask')
plt.subplot(133)
pred = model.predict(np.expand_dims(X_valid[image_number], 0))
pred = pred.squeeze(0)
pred = pred.squeeze(-1)
plt.imshow(pred, cmap='gray')
plt.title('Predicted ROI')
plt.savefig(f'{config_roi.TRAIN_PLOTS_PATH}/sanity_prediction.png')
plt.show()
##############################################################################
