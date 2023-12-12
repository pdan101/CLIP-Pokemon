import os
import clip
import torch
from torchvision.datasets import CIFAR100
from datasets import load_dataset
from pokemon_dataset import PokemonImageDataset

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

ds = PokemonImageDataset(train=False)
# import pdb; pdb.set_trace()
# print(ds['test'][0])
# Prepare the inputs
image, class_id = ds[295][0], ds[0][1][1]
print(f"True Class: {ds[295][1][0]}")
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat(
    [
        clip.tokenize(f"a photo of Pokemon named {c}")
        for c in list(ds.pokemon_names["name"])
    ]
).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{ds.pokemon_names['name'][index.item()]:>16s}: {100 * value.item():.2f}%")
