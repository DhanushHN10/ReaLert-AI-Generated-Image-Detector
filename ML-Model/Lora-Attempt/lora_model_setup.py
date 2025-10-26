from transformers import AutoModelForImageClassification
from peft import get_peft_model, LoraConfig
from torch import nn

model_name = "facebook/deit-base-patch16-224"

def download_lora_model(num_labels=6, lora_rank=8, alpha=32):
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )

    model.classifier = nn.Linear(model.classifier.in_features, num_labels)
    
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=alpha,
        target_modules=["query", "key", "value"],
        lora_dropout=0.1,
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    return model
