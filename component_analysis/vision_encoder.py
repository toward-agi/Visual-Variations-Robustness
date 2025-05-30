# models to be tested: 
# the vision encoder of Qwen2.5-VL-7b-instruct, Vila/Ovis(openai/clip-vit-large-patch14-336), InternVL(OpenGVLab/InternViT-300M-448px), meta-llama/Llama-3.2-11B-Vision 
#clip-vit-patch-32 already done
# tasks: object detection, direction prediction
# mode: zero-shot pred, linear probing

import datasets
import torch
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoProcessor
from tqdm import tqdm
import json
import argparse
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Setup argument parser
parser = argparse.ArgumentParser(description='Vision Encoder Analysis')
parser.add_argument('--task', type=str, choices=['object', 'direction'], required=True,
                   help='Task to perform: object or direction detection')
parser.add_argument('--model', type=str, default="openai/clip-vit-base-patch32",
                   help='Model to use for analysis')
parser.add_argument('--gpu', type=int, default=0,
                   help='GPU device number')
args = parser.parse_args()

# Common configurations
model_name = args.model
device = f"cuda:{args.gpu}"

if "clip" in model_name:
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.to(device)
elif "Qwen2-VL" in model_name:
    from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
    from qwen_vl_utils import process_vision_info
    processor = AutoProcessor.from_pretrained(model_name)
    # Use specific device instead of auto device_map
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype="auto",
        device_map={"": device}  # Map all components to the same device
    )
elif "InternViT" in model_name:
    model = AutoModel.from_pretrained("OpenGVLab/InternViT-300M-448px", trust_remote_code=True)
    processor = CLIPProcessor.from_pretrained("OpenGVLab/InternViT-300M-448px")
    model.to(device)
elif "Llama" in model_name:
    from transformers import MllamaForConditionalGeneration
    model = MllamaForConditionalGeneration.from_pretrained(
        "meta-llama/Llama-3.2-11B-Vision",
        device_map={"": device}  # Map all components to the same device
    )
    processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision")

results = []

# Task-specific configurations

base_path = "/data/zhiyuan/prjs/lvlm-bias-main/eval_dataset"
task2dataset = {
    "object": ["ratio-3_bg512-512-bgtype-colorshiba,dogwhite", "ratio-5_bg512-512-bgtype-colorshiba,dogwhite", 
               "ratio-2_bg512-512-bgtype-colorshiba,dogwhite", "ratio-10_bg512-512-bgtype-colorshiba,dogwhite"],
    "direction": ["ratio-2_bg512-512-bgtype-colorarrowwhite", "ratio-3_bg512-512-bgtype-colorarrowwhite", 
                 "ratio-5_bg512-512-bgtype-colorarrowwhite", "ratio-10_bg512-512-bgtype-colorarrowwhite"]
}

# Label lists
animal_list = ["antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar", 
               "butterfly", "cat", "caterpillar", "chimpanzee", "cockroach", "cow", 
               "coyote", "crab", "crow", "deer", "dog", "dolphin", "donkey", "dragonfly", 
               "duck", "eagle", "elephant", "flamingo", "fly", "fox", "goat", "goldfish", 
               "goose", "gorilla", "grasshopper", "hamster", "hare", "hedgehog", "hippopotamus", 
               "hornbill", "horse", "hummingbird", "hyena", "jellyfish", "kangaroo", "koala", "ladybugs", "leopard", "lion", "lizard", 
               "lobster", "mosquito", "moth", "mouse", "octopus", "okapi", "orangutan", "otter", "owl", "ox", "oyster", "panda", "parrot", 
               "pelecaniformes", "penguin", "pig", "pigeon", "porcupine", "possum", "raccoon", "rat", "reindeer", "rhinoceros", "sandpiper", 
               "seahorse", "seal", "shark", "sheep", "snake", "sparrow", "squid", "squirrel", "starfish", "swan", "tiger", "turkey", 
               "turtle", "whale", "wolf", "wombat", "woodpecker", "zebra"]


direction_list = ["The arrow is pointing right.", "The arrow is pointing upper right.", 
                 "The arrow is up.", "The arrow is pointing upper left.", "The arrow is pointing left.", 
                 "The arrow is pointing bottom left.", "The arrow is pointing down.", 
                 "The arrow is pointing bottom right."]

def process_object_detection(item):
    object_test = datasets.load_from_disk(f"{base_path}/object_dataset/{item}")
    object_test = object_test['test']
    object_test_results = []
    correct = 0
    total = 0

    for input_pair in tqdm(object_test):

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


        # Preprocess the inputs
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)


        inputs = processor(text=[text_prompt], images=[input_pair["image"]], 
                         return_tensors="pt", padding=True)
        # Move inputs to correct device
        # inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        inputs.to(device)
       
       
        outputs = model(**inputs)
        print(outputs.keys())
        logits_per_image = outputs.logits
        probs = logits_per_image.softmax(dim=1)
        pred_label = probs.argmax().item()
        print(pred_label)
        input_pair['output_text'] = animal_list[pred_label]
        if input_pair['output_text'] in input_pair['object_name']:
            correct += 1
        total += 1
        outpair = {k: v for k, v in input_pair.items() if k != "image"}
        outpair["predict_correct"] = input_pair['output_text'] in input_pair['object_name']
        object_test_results.append(outpair)

    with open(f"temp_results/object/vit_object_test_{item}.json", "w") as f:
        json.dump(object_test_results, f)
    return {"dataset": item, "accuracy": correct/total, "total_images": total}

def process_direction_detection(item):
    direction_test = datasets.load_from_disk(f"{base_path}/direction_dataset/{item}")
    direction_test = direction_test['test']
    direction_test_results = []
    correct = 0
    total = 0

    for input_pair in tqdm(direction_test):
        inputs = processor(text=direction_list, images=input_pair["image"], 
                         return_tensors="pt", padding=True)
        # Move inputs to correct device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        pred_label = probs.argmax().item()

        input_pair['output_text'] = pred_label
        if input_pair['output_text'] == input_pair['rotation_angle']//45:
            correct += 1
        total += 1
        outpair = {k: v for k, v in input_pair.items() if k != "image"}
        outpair["predict_correct"] = (input_pair['output_text'] == input_pair['rotation_angle']//45)
        direction_test_results.append(outpair)

    with open(f"temp_results/direction/vit_direction_test_{item}.json", "w") as f:
        json.dump(direction_test_results, f)
    return {"dataset": item, "accuracy": correct/total if total > 0 else 0, "total_images": total}

# Main execution
for item in task2dataset[args.task]:
    if args.task == "object":
        results.append(process_object_detection(item))
    else:  # direction
        results.append(process_direction_detection(item))
    print(f"Accuracy for {item}: {results[-1]['accuracy']}")
    print(f"Total number of images in {item}: {results[-1]['total_images']}")

# Calculate overall accuracy
overall_correct = sum(result["accuracy"] * result["total_images"] for result in results)
overall_total = sum(result["total_images"] for result in results)
print(overall_correct)
overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
print(f"Overall Accuracy: {overall_accuracy}")


# example usage
"""
python component_analysis/vision_encoder.py --task object --model openai/clip-vit-base-patch32 --gpu 6
python component_analysis/vision_encoder.py --task direction --model openai/clip-vit-base-patch32 --gpu 6
python component_analysis/vision_encoder.py --task object --model openai/clip-vit-large-patch14-336 --gpu 6
python component_analysis/vision_encoder.py --task direction --model openai/clip-vit-large-patch14-336 --gpu 6
python component_analysis/vision_encoder.py --task object --model Qwen/Qwen2-VL-7B-Instruct --gpu 6
python component_analysis/vision_encoder.py --task direction --model Qwen/Qwen2-VL-7B-Instruct --gpu 6
"""