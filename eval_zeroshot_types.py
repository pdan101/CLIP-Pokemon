import clip
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from pokemon_dataset import PokemonImageDataset
from get_types import get_all_types, pokemon_of_type


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

test_dataset = torch.load("./dataset.pth")
total_samples = len(test_dataset)
correct_predictions_type = 0
correct_predictions = 0
batch_size = 401  # Adjust the batch size as needed
imgs = []
names = []
name_idxs = []
type1_idx = []
type2_idx = []
potential_types = []

i = 0
for example in test_dataset:
    imgs.append(example[0])
    names.append(example[1][0])
    name_idxs.append(example[1][1])
    type1_idx.append(example[1][3])
    type2_idx.append(example[1][5])

types = get_all_types()
model.cuda().eval()
text_inputs = torch.cat(
    [clip.tokenize(f"this is photo of a {c} type pokemon") for c in types]
).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Loop through the test set in batches
for i in tqdm(range(0, total_samples, batch_size)):
    images = torch.stack([preprocess(image) for image in imgs[i : i + batch_size]]).to(
        device
    )

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(images)

    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)

    # Calculate similarity
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # Get the 9 top predictions for each sample in the batch
    values, indices = similarity.softmax(dim=-1).topk(17)

    # Check if the top predictions are correct
    for j in range(batch_size):
        if (type1_idx[i + j] in indices[j]) or (type2_idx[i + j] in indices[j]):
            correct_predictions_type += 1
        potential_types.append(indices[j])


# Calculate accuracy
accuracy = correct_predictions_type / total_samples
print(f"Accuracy: {accuracy * 100:.2f}%")


def flatten(l):
    return [item for sublist in l for item in sublist]


for i in tqdm(range(0, total_samples)):
    current_pokemon = []
    for t in potential_types[i]:
        current_pokemon.append(pokemon_of_type(types[t]))
    current_pokemon = list(set(flatten(current_pokemon)))
    text_inputs = torch.cat(
        [clip.tokenize(f"a photo of Pokemon named {c}") for c in current_pokemon]
    ).to(device)

    images = torch.stack([preprocess(imgs[i])]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        image_features = model.encode_image(images)

    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)

    # Calculate similarity
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    values, indices = similarity.softmax(dim=-1).topk(1)
    if names[i] == current_pokemon[indices[0].item()]:
        correct_predictions += 1
    # else:
    # print(i)
    # print(names[i])
    # print(current_pokemon[indices[0].item()])
    # print("\n")
accuracy = correct_predictions / total_samples
print(f"Accuracy: {accuracy * 100:.2f}%")
