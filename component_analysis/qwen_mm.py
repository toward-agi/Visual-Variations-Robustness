import datasets
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

animal_list = ["antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar", "butterfly", "cat", "caterpillar", "chimpanzee", "cockroach", "cow", "coyote", "crab", "crow", "deer", "dog", "dolphin", "donkey", "dragonfly", "duck", "eagle", "elephant", "flamingo", "fly", "fox", "goat", "goldfish", "goose", "gorilla", "grasshopper", "hamster", "hare", "hedgehog", "hippopotamus", "hornbill", "horse", "hummingbird", "hyena", "jellyfish", "kangaroo", "koala", "ladybugs", "leopard", "lion", "lizard", "lobster", "mosquito", "moth", "mouse", "octopus", "okapi", "orangutan", "otter", "owl", "ox", "oyster", "panda", "parrot", "pelecaniformes", "penguin", "pig", "pigeon", "porcupine", "possum", "raccoon", "rat", "reindeer", "rhinoceros", "sandpiper", "seahorse", "seal", "shark", "sheep", "snake", "sparrow", "squid", "squirrel", "starfish", "swan", "tiger", "turkey", "turtle", "whale", "wolf", "wombat", "woodpecker", "zebra"]

# model_name = "openai/clip-vit-large-patch14-336"
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
# model_name = "openai/clip-vit-base-patch32"
model_name = "Qwen/Qwen2-VL-7B-Instruct"
# model_name = 'OpenGVLab/InternViT-6B-448px-V1-5'
processor = AutoProcessor.from_pretrained(model_name)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto"  # Map all components to the same device
).cuda(7)
qwen2_vit = model.visual
qwen2_mm = model.Qwen2MLP

# model = Qwen2VisionTransformerPretrainedModel.from_pretrained(model_name)

results = []
# testing on our bias dataset
base_path = "/data/zhiyuan/prjs/lvlm-bias-main/eval_dataset"
task2dataset = {
    "object": ["ratio-2_bg512-512-bgtype-colorshiba,dogwhite", "ratio-3_bg512-512-bgtype-colorshiba,dogwhite", "ratio-5_bg512-512-bgtype-colorshiba,dogwhite", "ratio-10_bg512-512-bgtype-colorshiba,dogwhite"],
    "direction": ["ratio-2_bg512-512-bgtype-colorarrowwhite", "ratio-3_bg512-512-bgtype-colorarrowwhite", "ratio-5_bg512-512-bgtype-colorarrowwhite", "ratio-10_bg512-512-bgtype-colorarrowwhite"]
}
# load the object dataset
direction_list = ["right pointing arrow", "upper right pointing arrow", "upward pointing arrow", "upper left pointing arrow", "left pointing arrow", "downward pointing arrow", "lower left pointing arrow", "lower right pointing arrow"]

import tqdm
all_data = []
for item in task2dataset['direction']:
    dataset = datasets.load_from_disk(f"{base_path}/"+"direction_dataset"+f"/{item}")
    dataset = dataset['test']
    all_data.extend(dataset)

# randomly split the dataset into train and test
import random
random.seed(42)
random.shuffle(all_data)

# Calculate split index

total_size = len(all_data)
test_size = int(0.9 * total_size)
train_size = total_size - test_size

# Split the data
direction_train = all_data[:train_size]
direction_test = all_data[train_size:]

print(f"Training samples dir: {len(direction_train)}")
print(f"Testing samples dir: {len(direction_test)}")
    
obj_train = load_dataset("lucabaggi/animal-wildlife", split="test")
# only take the first 800 samples
obj_train = obj_train.select(range(700))
# Resize images to 512x512
obj_train = obj_train.map(lambda x: {"image": x["image"].convert("RGB").resize((512, 512)), "label": x["label"]})
object_test = []
for item in task2dataset['object']:

    dataset = datasets.load_from_disk(f"{base_path}/"+"object_dataset"+f"/{item}")
    # dataset = datasets.load_from_disk("/data/zhiyuan/prjs/lvlm-bias-main/eval_dataset/object_dataset/ratio-15_bg512-512-bgtype-colorshiba,dogwhite")
    dataset = dataset['test']
    object_test.extend(dataset)
    
    
    # Define linear probing model
# adding label "dog" to the object_test

for item in object_test:
    item["label"] = "dog"

print(f"Training samples obj: {len(obj_train)}")
print(f"Testing samples obj: {len(object_test)}")

device = "cuda:7"
class LinearProbe(nn.Module):
    def __init__(self, input_dim=1024, num_classes=8):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)
def direction_train_func():
    CLS_tokens = []
    for input_pair in tqdm.tqdm(direction_train):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[input_pair["image"]], padding=True, return_tensors="pt").to("cuda:7")
        outputs = qwen2_vit(inputs["pixel_values"].to(device), grid_thw = inputs["image_grid_thw"])
        # print(inputs.keys())
        #include the labels in the input
        # inputs = {k: v.to(torch.bfloat16) if torch.is_floating_point(v) else v for k, v in inputs.items()}
        # add the label

        # output_ids = model.generate(**inputs, max_new_tokens=128)
        # generated_ids = [
        #     output_ids[len(input_ids) :]
        #     for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        # ]
        # output_text = processor.batch_decode(
        #     generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        # )
        # get the CLS 
        # print(outputs.shape)
        CLS_token = outputs.mean(dim=0).cpu().detach()
        # print("CLS_token", CLS_token.shape)
        CLS_tokens.append(CLS_token.unsqueeze(dim=0).float()) # size 1*1024
        # print("CLS_token", CLS_token.shape)

    # Convert CLS tokens and labels to tensors
    CLS_tokens = torch.tensor(torch.cat(CLS_tokens, dim=0))
    train_labels = torch.tensor([data["rotation_angle"] // 45 for data in direction_train])
    print("CLS_tokens", CLS_tokens.shape)
    print("train_labels", train_labels.shape)


   
    linear_model = LinearProbe(input_dim=3584).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(linear_model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 50
    batch_size = 32
    train_dataset = TensorDataset(CLS_tokens, train_labels) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        linear_model.train()
        total_loss = 0
        for batch_tokens, batch_labels in train_loader:
            batch_tokens, batch_labels = batch_tokens.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = linear_model(batch_tokens)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

    # Evaluate on test set
    linear_model.eval()
    test_CLS_tokens = []
    for input_pair in tqdm.tqdm(direction_test):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[input_pair["image"]], padding=True, return_tensors="pt").to("cuda:7")
        outputs = qwen2_vit(inputs["pixel_values"].to(device), grid_thw = inputs["image_grid_thw"])

        # inputs = processor(images=input_pair["image"], return_tensors="pt").to(device)
        # inputs = {k: v.to(torch.bfloat16) if torch.is_floating_point(v) else v for k, v in inputs.items()}

        with torch.no_grad():
            # outputs = model(**inputs)
            outputs = qwen2_vit(inputs["pixel_values"].to(device), grid_thw = inputs["image_grid_thw"])
            CLS_token = outputs.mean(dim=0).cpu().detach()
            # print("CLS_token", CLS_token.shape)
            test_CLS_tokens.append(CLS_token.unsqueeze(dim=0).float())

    test_CLS_tokens = torch.tensor(torch.cat(test_CLS_tokens, dim=0)).to(device)
    test_labels = torch.tensor([data["rotation_angle"] // 45 for data in direction_test]).to(device)

    with torch.no_grad():
        test_outputs = linear_model(test_CLS_tokens)
        predictions = torch.argmax(test_outputs, dim=1)
        accuracy = (predictions == test_labels).float().mean()
        print(f"Test Accuracy: {accuracy.item():.4f}")

def object_train_func():
    CLS_tokens = []

    for input_pair in tqdm.tqdm(obj_train):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[input_pair["image"]], padding=True, return_tensors="pt").to("cuda:7")
        # inputs = processor(images=input_pair["image"], return_tensors="pt").to(device)
        # print(inputs.keys())
        #include the labels in the input
        inputs = {k: v.to(torch.bfloat16) if torch.is_floating_point(v) else v for k, v in inputs.items()}
        inputs = processor(text=[text_prompt], images=[input_pair["image"]], padding=True, return_tensors="pt").to("cuda:7")
        outputs = qwen2_vit(inputs["pixel_values"].to(device), grid_thw = inputs["image_grid_thw"])
       
        CLS_token = outputs.mean(dim=0).cpu().detach()
        # print("CLS_token", CLS_token.shape)
        CLS_tokens.append(CLS_token.unsqueeze(dim=0).float()) # size 1*1024

    # Convert CLS tokens and labels to tensors
    CLS_tokens = torch.tensor(np.concatenate(CLS_tokens, axis=0))
    train_labels = torch.tensor([data["label"] for data in obj_train])

    linear_model = LinearProbe(input_dim=3584, num_classes=90).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(linear_model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 50
    batch_size = 32
    train_dataset = TensorDataset(CLS_tokens, train_labels) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        linear_model.train()
        total_loss = 0
        for batch_tokens, batch_labels in train_loader:
            batch_tokens, batch_labels = batch_tokens.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = linear_model(batch_tokens)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

    # Evaluate on test set
    linear_model.eval()
    test_CLS_tokens = []
    print(len(object_test))
    obj_test = object_test[:800]
    for input_pair in tqdm.tqdm(object_test):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[input_pair["image"]], padding=True, return_tensors="pt").to("cuda:7")
        inputs = {k: v.to(torch.bfloat16) if torch.is_floating_point(v) else v for k, v in inputs.items()}
        with torch.no_grad():

            outputs = qwen2_vit(inputs["pixel_values"].to(device), grid_thw = inputs["image_grid_thw"])
            CLS_token = outputs.mean(dim=0).cpu().detach()
            # print("CLS_token", CLS_token.shape)
            test_CLS_tokens.append(CLS_token.unsqueeze(dim=0).float()) # size 1*1024
    
    test_CLS_tokens = torch.tensor(torch.cat(test_CLS_tokens, axis=0)).to(device)
    # test_labels = torch.tensor([data["label"] for data in object_test]).to(device)

    with torch.no_grad():
        test_outputs = linear_model(test_CLS_tokens)
        predictions = torch.argmax(test_outputs, dim=1)
        print(predictions.shape)
        predicted_animals = [animal_list[idx] for idx in predictions.cpu()]
        accuracy = torch.tensor([1 if animal == "dog" else 0 for animal in predicted_animals]).float().mean()
        print(f"Test Accuracy: {accuracy.item():.4f}")
# direction_train_func()
object_train_func()



