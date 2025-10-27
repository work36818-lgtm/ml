import numpy as np
import requests
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

img_size = 100
categories = ['healthy', 'diseased']

# Working image links
image_urls = [
    ("https://tse3.mm.bing.net/th/id/OIP.R3JS0blApMt6bgTTvIAGJAHaE5?pid=Api&P=0&h=180", "healthy"),
    ("https://tse2.mm.bing.net/th/id/OIP.wfhzV_eBziCdZkYnb6OfMQHaFj?pid=Api&P=0&h=180", "diseased")
]

data = []
labels = []

# ✅ Load images safely
for url, label in image_urls:
    try:
        print(f"Downloading: {url}")
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = img.resize((img_size, img_size))
        img_np = np.array(img)
        img_flat = img_np.flatten()
        data.append(img_flat)
        labels.append(categories.index(label))
        print(f"Processed: {label}")
    except Exception as e:
        print(f"Failed to process {url} → {e}")

# ✅ Convert to numpy arrays only if not empty
if len(data) > 0:
    data = np.array(data)
    labels = np.array(labels)

    # Train SVM
    clf = svm.SVC(kernel='linear')
    clf.fit(data, labels)
    predictions = clf.predict(data)

    # Show predictions with image
    for i, (url, actual_label) in enumerate(image_urls):
        img = Image.open(BytesIO(requests.get(url).content)).convert('RGB')
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Actual: {actual_label} | Predicted: {categories[predictions[i]]}")
        plt.show()

    print("Accuracy on training images:", accuracy_score(labels, predictions))
    print("Classification Report:\n", classification_report(labels, predictions, target_names=categories))
else:
    print("⚠ No images were processed. Please check URLs or internet connection.")
