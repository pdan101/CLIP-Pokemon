import os
import clip
import torch
from torch.utils.data import DataLoader
from pokemon_dataset import PokemonImageDataset

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

test_dataset = torch.load('./dataset.pth')
total_samples = len(test_dataset)
correct_predictions = 0
batch_size = 2  # Adjust the batch size as needed
imgs = []
names = []
name_idxs = []
for example in test_dataset:
    imgs.append(example[0])
    names.append(example[1][0])
    name_idxs.append(example[1][1])

# Loop through the test set in batches
for i in range(0, total_samples, batch_size):
    images = torch.stack([preprocess(image) for image in imgs[i:i+batch_size]]).to(device)
    text_inputs = torch.cat([
        clip.tokenize(f"a photo of Pokemon named {c}")
        for c in list(test_dataset.pokemon_names["name"])
    ]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(text_inputs)

    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Calculate similarity
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    # Get the top prediction for each sample in the batch
    top_predictions = similarity.argmax(dim=-1).tolist()

    # Check if the top predictions are correct
    for j in range(batch_size):
        if top_predictions[j] == name_idxs[i:i+j]:
            correct_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / total_samples
print(f"Accuracy: {accuracy * 100:.2f}%")