import glob
import os
import pandas as pd
from tensorflow.keras.models import load_model
from configs import config_roi
from utils.util import *

# Load model
model = load_model(config_roi.ROI_MODEL_PATH, compile=False)

# Load CSV with image files list
data = pd.read_csv(config_roi.IMAGES_CSV_PATH)
images_list = list(data['image_name'])

for img_path in glob.glob(os.path.join(config_roi.IMAGES_TO_EXTRACT_ROI_PATH, '*.png')):
    file_name = os.path.basename(img_path)
    if file_name in images_list:
        # Load and process image
        print(file_name)
        img_name, img_extension = os.path.splitext(file_name)
        img = read_image(img_path, aspect_ratio='square')  # (h,w,3)
        img_processed = preprocess_image(img, resize=True, size=224, equalize=True)  # (h,w,3)
        img_original = preprocess_image(img, resize=False, equalize=True)

        # Predict ROI
        pred = model.predict(np.expand_dims(img_processed, 0))
        pred = pred.squeeze(0)
        pred = pred.squeeze(-1)
        pred = cv2.resize(pred, (img_original.shape[:2]))

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
        for cnt in contours_sorted[:2]:
            x, y, w, h = cv2.boundingRect(cnt)
            # get the center and the radius
            cx = x + w // 2
            cy = y + h // 2
            cr = max(w, h) // 2

            roi = []
            roi = img_original[cy - cr:cy + cr, cx - cr:cx + cr]

            ###########
            if roi.size == 0:
                print(f'{img_name} is empty')
                break
            # Crop ROI from original resolution, resize to 512x512 and save
            # Left knee (located on the right hand side of the image) is flipped
            # vertically to match Right knee orientation
            # i.e. knee medial side is located on the left, lateral side on the right
            if cx >= img_original.shape[1] // 2:  # left knee
                cv2.imwrite(f'{config_roi.ROI_SAVE_PATH}/{img_name}_L{img_extension}', cv2.resize(roi, (512, 512)))
            else:  # right knee
                cv2.imwrite(f'{config_roi.ROI_SAVE_PATH}/{img_name}_R{img_extension}', cv2.flip(cv2.resize(roi, (512, 512)), 1))
            # Add bounding box to image
            cv2.rectangle(img_original, (cx - cr, cy - cr), (cx + cr, cy + cr), (0, 255, 0), 10)
        # save original image resized to 512x512 with ROI bounding boxes
        cv2.imwrite(f'{config_roi.ROI_VIZ_SAVE_PATH}/{img_name}_bbox{img_extension}', cv2.resize(img_original, (512, 512)))
