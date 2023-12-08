import os
import clip
import torch
from torch.utils.data import DataLoader
from pokemon_dataset import PokemonImageDataset
import torch.optim as optim
import matplotlib.pyplot as plt

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)

test_dataset = torch.load('./dataset.pth')
total_samples = len(test_dataset)
batch_size = 2  # Adjust the batch size as needed
imgs = []
names = []
name_idxs = []
for example in test_dataset:
    imgs.append(example[0])
    names.append(example[1][0])
    name_idxs.append(example[1][1])

loss_img = torch.nn.CrossEntropyLoss()
loss_txt = torch.nn.CrossEntropyLoss()

NUM_EPOCHS = 3
losses = []
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    # Loop through the test set in batches
    for i in range(0, total_samples, batch_size):
        optimizer.zero_grad()
        cur_end_idx = min(i+batch_size, len(test_dataset))

        images = torch.stack([preprocess(image) for image in imgs[i:cur_end_idx]]).to(device)
        text_inputs = torch.cat([
            clip.tokenize(f"a photo of Pokemon named {c}")
            for c in names[i:cur_end_idx]
        ]).to(device)

        logits_per_image, logits_per_text = model(images, text_inputs)
        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()

    # Print average loss for the epoch
    average_loss = epoch_loss / (total_samples / batch_size)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Average Loss: {average_loss}")
    losses.append(average_loss)

plt.plot(range(1, NUM_EPOCHS + 1), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Loss Over Epochs')
plt.show()