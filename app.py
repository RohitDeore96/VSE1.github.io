from flask import Flask, render_template, request
import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
import faiss
import os
import warnings

# Suppress duplicate OpenMP warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
DATASET_FOLDER = 'static/dataset_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

# Load model & index dataset once
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()
model = torch.nn.Sequential(*list(model.children())[:-1])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Index dataset images
image_paths = []
if os.path.exists(DATASET_FOLDER):
    image_paths = [os.path.join(DATASET_FOLDER, f) for f in os.listdir(DATASET_FOLDER) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

if not image_paths:
    print(f"⚠️ No images found in {DATASET_FOLDER}! Please add images.")

embeddings = []
for path in image_paths:
    img = Image.open(path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img_tensor).squeeze().numpy()
    embeddings.append(embedding)

if embeddings:
    embeddings = np.vstack(embeddings).astype('float32')
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)
else:
    faiss_index = None

@app.route('/', methods=['GET', 'POST'])
def index():
    query_image = None
    result_image = None
    similarity = None

    if request.method == 'POST' and faiss_index:
        file = request.files.get('query_image')
        if not file or file.filename == '':
            return render_template('index.html', query_image=None, result_image=None, similarity=None)

        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        query_img = Image.open(filepath).convert('RGB')
        query_tensor = transform(query_img).unsqueeze(0)
        with torch.no_grad():
            query_embedding = model(query_tensor).squeeze().numpy().astype('float32')

        query_embedding /= np.linalg.norm(query_embedding)
        distances, indices = faiss_index.search(np.expand_dims(query_embedding, axis=0), 1)

        matched_idx = indices[0][0]
        matched_path = image_paths[matched_idx]

        query_image = f"uploads/{filename}"
        result_image = f"dataset_images/{os.path.basename(matched_path)}"
        similarity = f"{distances[0][0]:.4f}"

    return render_template('index.html',
                            query_image=query_image,
                            result_image=result_image,
                            similarity=similarity)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
