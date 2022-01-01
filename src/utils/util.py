import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pydicom
from PIL import Image
from keras import backend as K
from tensorflow.keras.callbacks import History


def clahe(image):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.
    The image is divided into tiles of 8x8 pixel size and applies histogram equalization.
    To avoid noise amplification, the contrast correction is limited.
    """
    # Convert image to LAB Color so CLAHE can be applied to the luminance channel
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Splitting the LAB image to L (luminance), A and B channels, respectively
    l, a, b = cv2.split(lab_img)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(l)

    # Combine the CLAHE enhanced L-channel back with A and B channels
    updated_lab_img = cv2.merge((clahe_img, a, b))

    # Convert LAB image back to color (RGB)
    CLAHE_img = cv2.cvtColor(updated_lab_img, cv2.COLOR_LAB2BGR)

    return CLAHE_img


# def read_image(path):
#     image = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.uint8)
#     return image
def read_image(path, aspect_ratio=None):
    image = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.uint8)
    if aspect_ratio == 'square':
        image = reshape_square(image)
    return image


# def read_mask(path):
#     mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     return mask
def read_mask(path, aspect_ratio=None):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = np.expand_dims(mask, -1)
    if aspect_ratio == 'square':
        mask = reshape_square(mask)
    return mask


def preprocess_image(image, resize=False, size=None, equalize=True):
    # Reshape to square ratio
    # image_processed = reshape_square(image)

    # Image resize
    if resize:
        assert isinstance(size, int), 'size must be an integer'
        image = cv2.resize(image, (size, size))

    # CLAHE equalization
    if equalize:
        image_processed = clahe(image)

    # Normalize
    # image_processed = image_processed / 255.
    # image_processed = image_processed.astype(np.uint8)

    return image_processed


def preprocess_mask(mask, resize=False, size=None):
    # Add dimension for channel
    # mask_processed = np.expand_dims(mask, -1)

    # Reshape to square ratio
    # mask_processed = reshape_square(mask_processed)

    # Mask resize
    if resize:
        assert isinstance(size, int), 'size must be an integer'
        mask_processed = cv2.resize(mask, (size, size))

    # Normalize
    # mask_processed = mask_processed / 255.
    # mask_processed = mask_processed.astype(np.uint8)
    mask_processed = np.expand_dims(mask_processed, -1) / 255

    return mask_processed


def reshape_square(image):
    h, w, c = image.shape

    # Create a black image
    if h > w:
        x = y = w
        img_square = np.zeros((x, y, c), np.uint8)
    else:
        x = y = h
        img_square = np.zeros((x, y, c), np.uint8)

    # Fill square image template
    img_square = image[int((h - y) / 2):int(h - (h - y) / 2), int((w - x) / 2):int(w - (w - x) / 2), :]

    return img_square


def dicom_to_png(image_path):
    im = pydicom.dcmread(image_path)
    im = im.pixel_array.astype(float)
    im = (np.maximum(im, 0) / im.max()) * 255  # float64, pixel values (0-255)

    im = Image.fromarray(im)
    im = im.convert("L")
    return im


def get_roi_crop(image_original, image_processed, model):
    """
    Predicts ROI for left and right knee from processed image, and crops ROIs from original image to keep higher
    resolution. Left knee is flipped around vertical axis to match right knee orientation (i.e. knee medial side is
    located on the left, lateral side on the right ).
    Uses a trained model for ROI detection.
    Returns a dictionary with left and right knee ROI crop, and an image with bounding boxes.
    :param image_original: numpy array of shape [height, width, 3]
    :param image_processed: numpy array of shape [height, weight, 3]
    :param model: trained model for ROI detection
    :return: dictionary with left and right knee ROI crop, and image with bounding boxes
    """
    # Predict ROI
    pred = model.predict(np.expand_dims(image_processed, 0))
    pred = pred.squeeze(0)
    pred = pred.squeeze(-1)
    pred = cv2.resize(pred, (image_original.shape[:2]))

    # Obtain contours of ROI
    ret, thresh = cv2.threshold(pred, 0.5, 1, 0)
    thresh = 255 * thresh  # scale by 255
    thresh = thresh.astype(np.uint8)
    contours, hierarchy = cv2.findContours(thresh, mode=cv2.cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour by area
    areas = [cv2.contourArea(c) for c in contours]
    contours_sorted = [x for _, x in sorted(zip(areas, contours), reverse=True)]
    contours_max = contours_sorted[:1]

    # Make image uint8
    pred *= 255
    pred = pred.astype(np.uint8)

    # Obtain center and size of square bounding box that contains ROI
    images = {}
    for cnt in contours_sorted[:2]:
        x, y, w, h = cv2.boundingRect(cnt)
        # get the center and the radius
        cx = x + w // 2
        cy = y + h // 2
        cr = max(w, h) // 2

        roi = []
        roi = image_original[cy - cr:cy + cr, cx - cr:cx + cr]

        # Crop ROI from original resolution, resize to 512x512 and save
        # Left knee (located on the right hand side of the image) is flipped
        # vertically to match Right knee orientation
        # i.e. knee medial side is located on the left, lateral side on the right
        if cx >= image_original.shape[1] // 2:  # left knee
            images['left'] = cv2.resize(roi, (512, 512))
        else:  # right knee
            images['right'] = cv2.flip(cv2.resize(roi, (512, 512)), 1)
        # Add bounding box to image
        cv2.rectangle(image_original, (cx - cr, cy - cr), (cx + cr, cy + cr), (0, 255, 0), 10)
    # save original image resized to 512x512 with ROI bounding boxes
    images['bbox'] = cv2.resize(image_original, (512, 512))

    return images


def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed


class LossLearningRateScheduler(History):
    """
    base_lr: the starting learning rate
    lookback_epochs: the number of epochs in the past to compare with the loss function at the current epoch to determine if progress is being made.
    decay_threshold / decay_multiple: if loss function has not improved by a factor of decay_threshold * lookback_epochs, then decay_multiple will be applied to the learning rate.
    spike_epochs: list of the epoch numbers where you want to spike the learning rate.
    spike_multiple: the multiple applied to the current learning rate for a spike.
    """

    def __init__(self, base_lr, lookback_epochs, spike_epochs=None, spike_multiple=10, decay_threshold=0.002,
                 decay_multiple=0.1, loss_type='val_loss'):
        super(LossLearningRateScheduler, self).__init__()
        self.base_lr = base_lr
        self.lookback_epochs = lookback_epochs
        self.spike_epochs = spike_epochs
        self.spike_multiple = spike_multiple
        self.decay_threshold = decay_threshold
        self.decay_multiple = decay_multiple
        self.loss_type = loss_type

    def on_epoch_begin(self, epoch, logs=None):
        if len(self.epoch) > self.lookback_epochs:
            current_lr = K.get_value(self.model.optimizer.lr)
            target_loss = self.history[self.loss_type]
            loss_diff = target_loss[-int(self.lookback_epochs)] - target_loss[-1]

            if loss_diff <= np.abs(target_loss[-1]) * (self.decay_threshold * self.lookback_epochs):
                print(' '.join(
                    ('Changing learning rate from', str(current_lr), 'to',
                     str(round(current_lr * self.decay_multiple, 6)))))
                K.set_value(self.model.optimizer.lr, round(current_lr * self.decay_multiple, 6))
                current_lr = current_lr * self.decay_multiple
            else:
                print(' '.join(('Learning rate:', str(current_lr))))

            if self.spike_epochs is not None and len(self.epoch) in self.spike_epochs:
                print(' '.join(
                    ('Spiking learning rate from', str(current_lr), 'to',
                     str(round(current_lr * self.spike_multiple, 6)))))
                K.set_value(self.model.optimizer.lr, round(current_lr * self.spike_multiple, 6))
        else:
            print(' '.join(('Learning rate:', str(self.base_lr))))
            K.set_value(self.model.optimizer.lr, self.base_lr)

        return K.get_value(self.model.optimizer.lr)

