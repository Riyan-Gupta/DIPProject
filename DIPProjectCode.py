import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from skimage.feature import graycomatrix, graycoprops
from joblib import Parallel, delayed
import seaborn as sns
import lightgbm as lgb
from sklearn.cluster import KMeans

# Path to the dataset
dataset_path = 'C:/Users/riyan/Downloads/CCMT_Final Dataset'

# Function to load and preprocess images with healthy/diseased distinction
def load_and_preprocess_images_with_subfolders(folder, label_map, img_size=(256, 256), max_images_per_folder=400):
    images = []
    labels = []
    for subfolder, label in label_map.items():
        subfolder_path = os.path.join(folder, subfolder)
        for i, filename in enumerate(os.listdir(subfolder_path)):
            if i >= max_images_per_folder:
                break
            img_path = os.path.join(subfolder_path, filename)
            if os.path.isfile(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img_resized = cv2.resize(img, img_size)
                    img_normalized = img_resized / 255.0
                    images.append(img_normalized)
                    labels.append(label)
    return np.array(images), np.array(labels)

# Load images for all categories with healthy and diseased subfolders
categories = ['cashew', 'cassava', 'maize', 'tomato']
images = []
labels = []

# Map subfolder names to labels
# Healthy: 0, Diseased: 1
subfolder_label_map = {'healthy': 0, 'diseased': 1}

for category in categories:
    folder_path = os.path.join(dataset_path, category)
    imgs, lbls = load_and_preprocess_images_with_subfolders(folder_path, subfolder_label_map)
    images.append(imgs)
    labels.append(lbls)

# Combine images and labels from all categories
images = np.vstack(images)
labels = np.hstack(labels)

print(f"Total images: {len(images)}, Total labels: {len(labels)}")

# Display Preprocessed Image
img_index = 0  # Change index as needed
original_img = images[img_index]

plt.figure(figsize=(6, 6))
plt.imshow(original_img)
plt.title(f"Preprocessed Image at Index {img_index}")
plt.axis('off')
plt.show()

# Function to extract features from a single image
def extract_features_single(img):
    img = (img * 255).astype(np.uint8)
    hist = cv2.calcHist([img], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_img, distances=[2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    energy = graycoprops(glcm, 'energy').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    return np.hstack([hist, contrast, correlation, energy, homogeneity])

# Extract and Display Features for Single Image
single_img_features = extract_features_single(images[img_index])
print(f"Feature vector for image at index {img_index}:")
print(single_img_features)

plt.figure(figsize=(10, 4))
plt.plot(single_img_features, marker='o')
plt.title(f"Feature Vector for Image at Index {img_index}")
plt.xlabel("Feature Index")
plt.ylabel("Feature Value")
plt.show()

# Parallel Feature Extraction
features = Parallel(n_jobs=-1)(delayed(extract_features_single)(img) for img in images)

# Standardize Features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

# PCA for dimensionality reduction
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Hybrid 1: Voting Classifier
rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
et_model = ExtraTreesClassifier(n_estimators=300, random_state=42)

voting_model = VotingClassifier(
    estimators=[('rf', rf_model), ('gb', gb_model), ('et', et_model)],
    voting='soft'
)
voting_model.fit(X_train_pca, y_train)
y_pred_voting = voting_model.predict(X_test_pca)
print("Hybrid 1: Voting Classifier")
print("Accuracy:", accuracy_score(y_test, y_pred_voting))
print(classification_report(y_test, y_pred_voting, target_names=['Healthy', 'Diseased']))

# Confusion Matrix for Voting Classifier
cm_voting = confusion_matrix(y_test, y_pred_voting)
plt.figure(figsize=(6, 6))
sns.heatmap(cm_voting, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Diseased'], yticklabels=['Healthy', 'Diseased'])
plt.title('Confusion Matrix - Voting Classifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Hybrid 2: PCA + RFE
rfe_model = RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=50)
X_train_rfe = rfe_model.fit_transform(X_train_pca, y_train)
X_test_rfe = rfe_model.transform(X_test_pca)

rf_rfe_model = RandomForestClassifier(n_estimators=300, random_state=42)
rf_rfe_model.fit(X_train_rfe, y_train)
y_pred_rfe = rf_rfe_model.predict(X_test_rfe)
print("Hybrid 2: PCA + RFE")
print("Accuracy:", accuracy_score(y_test, y_pred_rfe))
print(classification_report(y_test, y_pred_rfe, target_names=['Healthy', 'Diseased']))

# Confusion Matrix for PCA + RFE
cm_rfe = confusion_matrix(y_test, y_pred_rfe)
plt.figure(figsize=(6, 6))
sns.heatmap(cm_rfe, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Diseased'], yticklabels=['Healthy', 'Diseased'])
plt.title('Confusion Matrix - PCA + RFE')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Hybrid 3: Augmented Features + LightGBM
def augment_features(img):
    gray_img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray_img)
    variance = np.var(gray_img)
    glcm_features = extract_features_single(img)
    return np.hstack([glcm_features, mean, variance])

augmented_features = Parallel(n_jobs=-1)(delayed(augment_features)(img) for img in images)
augmented_features = scaler.fit_transform(augmented_features)
X_train_aug, X_test_aug, y_train_aug, y_test_aug = train_test_split(augmented_features, labels, test_size=0.2, random_state=42, stratify=labels)

lgb_model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=42)
lgb_model.fit(X_train_aug, y_train_aug)
y_pred_lgb = lgb_model.predict(X_test_aug)
print("Hybrid 3: Augmented Features + LightGBM")
print("Accuracy:", accuracy_score(y_test_aug, y_pred_lgb))
print(classification_report(y_test_aug, y_pred_lgb, target_names=['Healthy', 'Diseased']))

# Confusion Matrix for Augmented Features + LightGBM
cm_lgb = confusion_matrix(y_test_aug, y_pred_lgb)
plt.figure(figsize=(6, 6))
sns.heatmap(cm_lgb, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Diseased'], yticklabels=['Healthy', 'Diseased'])
plt.title('Confusion Matrix - Augmented Features + LightGBM')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Hybrid 4: Clustering + Classification
kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans.fit_predict(features)
hybrid_features = np.hstack([features, cluster_labels.reshape(-1, 1)])
X_train_clust, X_test_clust, y_train_clust, y_test_clust = train_test_split(hybrid_features, labels, test_size=0.2, random_state=42, stratify=labels)

rf_clustered = RandomForestClassifier(n_estimators=300, random_state=42)
rf_clustered.fit(X_train_clust, y_train_clust)
y_pred_clustered = rf_clustered.predict(X_test_clust)
print("Hybrid 4: Clustering + Classification")
print("Accuracy:", accuracy_score(y_test_clust, y_pred_clustered))
print(classification_report(y_test_clust, y_pred_clustered, target_names=['Healthy', 'Diseased']))

# Confusion Matrix for Clustering + Classification
cm_cluster = confusion_matrix(y_test_clust, y_pred_clustered)
plt.figure(figsize=(6, 6))
sns.heatmap(cm_cluster, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Diseased'], yticklabels=['Healthy', 'Diseased'])
plt.title('Confusion Matrix - Clustering + Classification')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

