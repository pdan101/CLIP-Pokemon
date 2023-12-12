import os
import numpy as np
import clip
import torch
from torch.utils.data import DataLoader
from pokemon_dataset import PokemonImageDataset

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

test_dataset = torch.load('./dataset.pth')
total_samples = len(test_dataset)
batch_size = 2  # Adjust the batch size as needed

imgs = []
name_idxs = []
for example in test_dataset:
    imgs.append(example[0])
    name_idxs.append(example[1][1])

token_arr = []
for name in test_dataset.pokemon_names['name'].tolist():
    type1 = test_dataset.pokemon_types[test_dataset.pokemon_types["name"] == name]["type1"].item()
    temp = test_dataset.pokemon_types[test_dataset.pokemon_types["name"] == name]["type2"].item()
    type2 = temp if (type(temp) == str) else type1
    token_arr.append((name,type1,type2))

def evaluate_caption(text_inputs):
    correct_predictions = 0

    model.eval()
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Loop through the test set in batches
    for i in range(0, total_samples, batch_size):
        images = torch.stack([preprocess(image) for image in imgs[i:i+batch_size]]).to(device)
        
        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(images)
            
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Calculate similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        # Get the top prediction for each sample in the batch
        top_predictions = similarity.softmax(dim=-1).topk(1)

        # Check if the top predictions are correct
        for j in range(batch_size):
            if name_idxs[i + j] in top_predictions[1][j]:
                correct_predictions += 1
        
    # Calculate accuracy
    accuracy = correct_predictions / total_samples
    print(f"Accuracy: {accuracy * 100:.2f}%")

print("A photo of Pokemon named (name)")
evaluate_caption(torch.cat([
    clip.tokenize(f"A photo of Pokemon named {name}")
    for name, type1, type2 in token_arr
    ]).to(device))

print("A photo of Pokemon with primary type (type1)")
evaluate_caption(torch.cat([
    clip.tokenize(f"A photo of Pokemon with primary type {type1}")
    for name, type1, type2 in token_arr
    ]).to(device))

print("A photo of Pokemon with primary type (type2)")
evaluate_caption(torch.cat([
    clip.tokenize(f"A photo of Pokemon with secondary type {type2}")
    for name, type1, type2 in token_arr
    ]).to(device))

print("A photo of Pokemon named (name) with primary type (type1)")
evaluate_caption(torch.cat([
    clip.tokenize(f"A photo of Pokemon named {name} with primary type {type1}")
    for name, type1, type2 in token_arr
    ]).to(device))

print("A photo of Pokemon named (name) with secondary type (type2)")
evaluate_caption(torch.cat([
    clip.tokenize(f"A photo of Pokemon named {name} with secondary type {type2}")
    for name, type1, type2 in token_arr
    ]).to(device))

print("A photo of Pokemon with types (type1) and (type2)")
evaluate_caption(torch.cat([
    clip.tokenize(f"A photo of Pokemon with types {type1} and {type2}")
    for name, type1, type2 in token_arr
    ]).to(device))

print("A photo of Pokemon named (name) with types (type1) and (type2)")
evaluate_caption(torch.cat([
    clip.tokenize(f"A photo of Pokemon named {name} with types {type1} and {type2}")
    for name, type1, type2 in token_arr
    ]).to(device))