from transformers import AutoModelForImageClassification
from peft import PeftModel
import torch
import os 

print("Loading base model (facebook/deit-base-patch16-224)...")
base_model = AutoModelForImageClassification.from_pretrained(
    "facebook/deit-base-patch16-224", 
    num_labels=6,
    ignore_mismatched_sizes=True  
)


adapter_save_path = "./AI-Generated-Image-Detector-Model"

print(f"Loading LoRA adapters from {adapter_save_path}...")
lora_model = PeftModel.from_pretrained(
    base_model,
    adapter_save_path 
)

print("Merging model adapters...")
merged_model = lora_model.merge_and_unload()
print("Merge complete.")


print("Loading trained classifier head weights...")
try:
    head_weights_path = f"{adapter_save_path}/classifier_head.pth"
    head_weights = torch.load(head_weights_path)
    merged_model.classifier.load_state_dict(head_weights)
    print("Successfully loaded and applied classifier head.")
except FileNotFoundError:
    print(f"--- ERROR! ---")
    print(f"Could not find the file: {head_weights_path}")
    print(f"Please make sure you re-ran train.py successfully first!")
    print("Aborting merge.")
    exit() 
except Exception as e:
    print(f"An error occurred loading the classifier head: {e}")
    exit()


save_path = "./FinalMergedModels/ai_generated_image_detector_merged_model"
print(f"Saving final merged model to {save_path}...")
os.makedirs(os.path.dirname(save_path), exist_ok=True) 
merged_model.save_pretrained(save_path)

print(f"Merged model saved successfully at {save_path}")
print("You are all set! You can now run test.py and inference.py 🚀")