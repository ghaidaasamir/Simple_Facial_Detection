# Face Detection using HOG and Sliding Window

Face detection pipeline using HOG features and a sliding window approach. The overall process includes data preparation, feature extraction, classifier training, and, applying a sliding window detector on a test image.

---

## 1. Dataset Preparation

- **Positive Patches:**  
  LFW dataset from scikit-learn. The images from this dataset (faces) as the positive examples.
  
- **Negative Patches:**  
  A collection of diverse images (e.g., 'camera', 'text', 'coins', etc.) from the scikit-image data module to extract negative patches. 

  For each image, patches are extracted at three different scales (0.5, 1.0, and 2.0) using scikit-learn's PatchExtractor.

---

## 2. Feature Extraction

- **HOG Features:**  
  For both positive and negative patches, HOG is computed.  
  HOG captures edge and gradient structures, for face detection.

---

## 3. Classifier Training

- **Feature Vector:**  
  The HOG features from all patches are combined into a training set.  
  A corresponding label array is created, where positive patches (faces) are labeled as 1 and negative patches as 0.

- **Model Training:**  
  - Gaussian Naive Bayes classifier with cross-validation.
  - Linear Support Vector Classifier (LinearSVC) is tuned using GridSearchCV to find the best regularization parameter (C).  

---

## 4. Detection with Sliding Window

- **Test Image Preparation:**  
  A test image is loaded , converted to grayscale, and rescaled.
  
- **Sliding Window Approach:**  
  A sliding window scans the test image in small steps. For each window (patch), HOG features are computed and passed through the trained classifier.
  
- **Detection Visualization:**  
  Patches classified as positive are marked with yellow rectangles on the test image.  

---

## How to Run

1. **Install Dependencies:**  
   - `numpy`
   - `scikit-learn`
   - `scikit-image`
   - `matplotlib`

2. **Main Script:**  
   Run simple_facial_detection.ipynb for the entire pipeline.  
  
