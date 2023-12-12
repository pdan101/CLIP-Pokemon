import pandas as pd


# Helper functions for accessing pokemon and their types
def pokemon_of_type(type):
    typedf = pd.read_csv("./pokemon_data/pokemon_types.csv")
    names = set(pd.read_csv("./pokemon_data/ds_labels.csv")["name"])
    return list(
        (set(typedf[(typedf["type1"] == type) | (typedf["type2"] == type)]["name"]))
        & names
    )


def get_all_types():
    typedf = pd.read_csv("./pokemon_data/pokemon_types.csv")
    primary = typedf["type1"].unique()
    return list(primary)


def type_pokemon(name):
    typedf = pd.read_csv("./pokemon_data/pokemon_types.csv")
    entry = typedf[(typedf["name"] == name)]
    return (entry["type1"].item(), entry["type2"].item())
