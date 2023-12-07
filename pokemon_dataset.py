import os
import pandas as pd
import torch
from torchvision.io import read_image
from datasets import load_dataset
from torch.utils.data import Dataset


class PokemonImageDataset(Dataset):
    def __init__(self, train=True):
        ds = load_dataset("fcakyon/pokemon-classification", name="full")
        ds = ds.with_format("torch")
        pokemon_names = pd.read_csv("./pokemon_data/ds_labels.csv")
        pokemon_types = pd.read_csv("./pokemon_data/pokemon_types.csv")
        self.pokemon_names = pokemon_names
        self.pokemon_types = pokemon_types
        if train:
            self.img_dir = ds["train"]["image_file_path"]
            self.img_label = ds["train"]["labels"]
        else:
            self.img_dir = ds["test"]["image_file_path"]
            self.img_label = ds["test"]["labels"]

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img_path = self.img_dir[idx]
        image = read_image(img_path)
        name_idx = self.img_label[idx].item()
        name = self.pokemon_names.iloc[name_idx]["name"]
        type1 = self.pokemon_types[self.pokemon_types["name"] == name][
            "type1_labels"
        ].item()
        type2 = self.pokemon_types[self.pokemon_types["name"] == name][
            "type2_labels"
        ].item()
        return image, (name_idx, type1, type2)


if __name__ == "__main__":
    data = PokemonImageDataset()
    train_features, train_labels = next(iter(data))
    print(train_features)
    print(train_labels)
    print("done")
