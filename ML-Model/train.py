import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_scheduler, AutoModelForImageClassification
from data import get_dataLoaders, load_dataset_clean
from model.optimizer import get_optimizer
from model.metrics import compute_metrics, accuracy_metric
from tqdm import tqdm
import os

DEBUG = True 

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if DEBUG:
        if device.type == "cuda":
            print("Using CUDA")
            print("CUDA device count:", torch.cuda.device_count())
            print("Current CUDA device:", torch.cuda.current_device())
            print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
        else:
            print("CUDA not available, using CPU")

    BATCH_SIZE = 26
    EPOCHS = 5
    SAVE_DIR = "Final-AI-Generated-Image-Detector-Model"
    RESUME_TRAINING_DIR = "New-AI-Generated-Image-Detector-Model"

    train_ds, validation_ds = load_dataset_clean("HPAI-BSC/SuSy-Dataset")
    train_loader, validation_loader = get_dataLoaders(train_ds, validation_ds, batch_size=BATCH_SIZE)


    # print("Loading SMALL model (facebook/deit-small-patch16-224) for full fine-tuning...")
    # model = AutoModelForImageClassification.from_pretrained(
    #     "facebook/deit-small-patch16-22", 
    #     num_labels=6,                
    #     ignore_mismatched_sizes=True 
    # ).to(device)

    print(f"Loading partially trained model for resumption from {RESUME_TRAINING_DIR} for full fine-tuning...")
    model = AutoModelForImageClassification.from_pretrained(
        RESUME_TRAINING_DIR,
    ).to(device)

    if DEBUG:
        print("Model device:", next(model.parameters()).device)


    
    optimizer = get_optimizer(model)

    criterion = nn.CrossEntropyLoss() 
    
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=EPOCHS * len(train_loader)
    )

    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        train_acc = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()
            inputs = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_acc += accuracy_metric(outputs, labels).item()
        
        train_acc /= len(train_loader)
        print(f"Epoch {epoch+1}, Training Accuracy: {train_acc:.4f}")

        model.eval()
        val_acc = 0
        with torch.no_grad():
            for batch in tqdm(validation_loader, desc=f"Validation Epoch {epoch+1}"):
                inputs = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                logits = model(inputs).logits
                val_acc += accuracy_metric(logits, labels).item()
        val_acc /= len(validation_loader)
        print(f"Epoch {epoch+1}, Validation Accuracy: {val_acc:.4f}")


    print(f"Saving fully fine-tuned model to {SAVE_DIR}...")
    model.save_pretrained(SAVE_DIR)
    

    
    print(f"Full model saved to {SAVE_DIR}")