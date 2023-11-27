import pandas as pd

pokemon = pd.read_csv("./pokemon_data/pokemon.csv")
labels = pokemon[["name", "type1", "type2"]]
labels["type1_labels"], label_index = pd.factorize(labels["type1"])


def enumerate_types(x):
    try:
        val = list(label_index).index(x)
    except ValueError:
        val = -1
    return val


labels["type2_labels"] = labels["type2"].apply(enumerate_types)
labels.to_csv("./pokemon_data/pokemon_types.csv", index=False)
