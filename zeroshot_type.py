import os
import clip
import torch
from torchvision.datasets import CIFAR100
from datasets import load_dataset
from pokemon_dataset import PokemonImageDataset
from get_types import get_all_types


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

types = get_all_types()

ds = torch.load("./dataset.pth")
# import pdb; pdb.set_trace()
# print(ds['test'][0])
# Prepare the inputs

index_ex = 500
image = ds[index_ex][0]
print(f"True Class: {ds[index_ex][1][0]}")
print(f"True Class: {ds[index_ex][1][2]}")
print(f"True Class: {ds[index_ex][1][4]}")
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat(
    [clip.tokenize(f"this is a {c} type pokemon") for c in types]
).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(18)

# Print the result
print("\nTop predictions:\n")
sum = 0
for val in values:
    sum += val.item()
for value, index in zip(values, indices):
    print(f"{types[index.item()]:>16s}: {100 * value.item():.2f}%")

print(str(sum) + "\n")

if (ds[index_ex][1][3] in indices[0]) | (ds[index_ex][1][5] in indices[0]):
    print("right")
else:
    print("wrong")
