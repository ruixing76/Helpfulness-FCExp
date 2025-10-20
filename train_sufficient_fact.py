import os
import torch
import logging
import argparse
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoConfig, AutoTokenizer, AutoModel, Trainer, TrainingArguments, PreTrainedModel
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from utils.config import PLM_NAME_MAPPING
from utils.data_io import read_jsonl

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Label mapping for sufficient fact prediction
SUFFICIENT_FACT_LABELS = {
    "NOT ENOUGH": 0,  # Evidence is not sufficient
    "ENOUGH -- IRRELEVANT": 1,    # Evidence is sufficient (supports claim)
    "ENOUGH -- REPEATED": 1      # Evidence is sufficient (refutes claim)
}

class SufficientFactDataset(Dataset):
    def __init__(self, dataset=None, tokenizer=None, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = dataset
          
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        claim = item["claim"]
        
        # Concatenate all evidence pieces
        evidence_texts = []
        for evidence_pair in item["evidence"]:
            if len(evidence_pair) >= 2:
                title, text = evidence_pair[0], evidence_pair[1]
                evidence_texts.append(f"{title}: {text}")
        
        evidence = " ".join(evidence_texts)
        
        # Create input text: Claim [SEP] Evidence
        text = f"Claim: {claim} [SEP] Evidence: {evidence}"
        
        # Tokenize the text
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None
        )
        
        # Convert label_after to binary sufficient/not sufficient
        inputs["labels"] = SUFFICIENT_FACT_LABELS[item["label_after"]]
        
        return inputs

class SufficientFactClassifier(PreTrainedModel):
    def __init__(self, config=None):
        super().__init__(config)
        self.config = config
        
        # Load pre-trained model
        self.model = AutoModel.from_pretrained(self.config._name_or_path)
        self.dropout = nn.Dropout(0.1)
        
        # Get hidden size from model config
        hidden_size = self.config.hidden_size
        
        # Binary classification head
        self.classifier = nn.Linear(hidden_size, 2)
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        # Get model outputs
        if 'ModernBERT' in self.config._name_or_path:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids 
            )
        
        # Get sequence output and apply dropout
        sequence_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token representation
        sequence_output = self.dropout(sequence_output)
        
        # Classification
        logits = self.classifier(sequence_output)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class DataCollatorForSufficientFact:
    def __init__(self, tokenizer, padding=True, max_length=512, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        batch = {
            'input_ids': torch.tensor([f['input_ids'] for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f['attention_mask'] for f in features], dtype=torch.long),
            'labels': torch.tensor([f['labels'] for f in features], dtype=torch.long)
        }
        
        # Add token_type_ids if available
        if 'token_type_ids' in features[0]:
            batch['token_type_ids'] = torch.tensor([f['token_type_ids'] for f in features], dtype=torch.long)
            
        return batch

# Metrics computation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    # Overall metrics
    accuracy = accuracy_score(labels, preds)
    f1_weighted = f1_score(labels, preds, average='weighted')
    precision_weighted = precision_score(labels, preds, average='weighted')
    recall_weighted = recall_score(labels, preds, average='weighted')
    
    # Per-class metrics
    f1_per_class = f1_score(labels, preds, average=None)
    precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
    recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
    
    # Create class names for better readability
    class_names = ["NOT_ENOUGH", "ENOUGH"]
    
    # Print detailed classification report
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(labels, preds, target_names=class_names, digits=4))
    
    # Print per-class metrics
    print("\nPER-CLASS METRICS:")
    print("-" * 40)
    for i, class_name in enumerate(class_names):
        if i < len(f1_per_class):
            print(f"{class_name:12} - Precision: {precision_per_class[i]:.4f}, Recall: {recall_per_class[i]:.4f}, F1: {f1_per_class[i]:.4f}")
    
    # Build metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'f1': f1_weighted,
        'precision': precision_weighted,
        'recall': recall_weighted,
    }
    
    # Add per-class metrics to the returned dictionary
    for i, class_name in enumerate(class_names):
        if i < len(f1_per_class):
            metrics[f'f1_{class_name.lower()}'] = f1_per_class[i]
            metrics[f'precision_{class_name.lower()}'] = precision_per_class[i]
            metrics[f'recall_{class_name.lower()}'] = recall_per_class[i]
    
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='./data/fc_dataset/sufficient_fact/train.jsonl')
    parser.add_argument('--test_data', type=str, default='./data/fc_dataset/sufficient_fact/test.jsonl')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    parser.add_argument('--model_name', type=str, default='roberta-base')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='./output-sufficient-fact')
    parser.add_argument('--save_dir', type=str, default='./save-sufficient-fact')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--resume_from_checkpoint', type=bool, default=False)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--report_to', type=str, default='wandb', choices=['wandb', 'none'])
    parser.add_argument('--run_name', type=str, default='')
    args = parser.parse_args()

    # Get model name from mapping
    model_name = PLM_NAME_MAPPING.get(args.model_name, args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    config._name_or_path = model_name
    model = SufficientFactClassifier(config=config)

    # Load datasets
    print("Loading the dataset...")
    train_data = read_jsonl(args.train_data)
    test_data = read_jsonl(args.test_data)
    
    train_dataset = SufficientFactDataset(dataset=train_data, tokenizer=tokenizer, max_length=args.max_length)
    test_dataset = SufficientFactDataset(dataset=test_data, tokenizer=tokenizer, max_length=args.max_length)

    # Print dataset statistics
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Count label distribution
    train_labels = [item["label_after"] for item in train_data]
    test_labels = [item["label_after"] for item in test_data]
    
    print(f"Train label distribution: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
    print(f"Test label distribution: {dict(zip(*np.unique(test_labels, return_counts=True)))}")

    if args.mode == 'train':
        print(f"Training the {args.model_name}...")
        training_args = TrainingArguments(
            output_dir=f"{args.output_dir}/{args.model_name}",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            
            save_strategy="epoch",
            eval_strategy="epoch",
            
            logging_dir=args.log_dir,
            logging_strategy="steps",
            logging_steps=100,
            log_level="info",

            weight_decay=0.01,
            
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            
            load_best_model_at_end=True,
            resume_from_checkpoint=args.resume_from_checkpoint,
            save_safetensors=True,
            report_to=args.report_to,
            run_name=args.run_name,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorForSufficientFact(tokenizer, max_length=args.max_length),
        )
        
        trainer.train()
        trainer.save_model(f"{args.save_dir}/{args.model_name}")

        # Final evaluation on test set
        print(f"Final evaluation on test set...")
        eval_results = trainer.evaluate(test_dataset)
        print(eval_results)

    elif args.mode == 'eval':
        print(f"Evaluating the {args.model_name}...")
        
        # Load the saved model for evaluation
        saved_model_path = f"{args.save_dir}/{args.model_name}"
        if os.path.exists(saved_model_path):
            print(f"Loading model from {saved_model_path}")
            try:
                # Load the saved model
                saved_config = AutoConfig.from_pretrained(saved_model_path)
                saved_tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
                saved_model = SufficientFactClassifier.from_pretrained(saved_model_path, config=saved_config)
                
                # Update tokenizer and recreate test dataset if needed
                test_dataset = SufficientFactDataset(dataset=test_data, tokenizer=saved_tokenizer, max_length=args.max_length)
                
                print(f"✓ Successfully loaded saved model from {saved_model_path}")
                model_to_eval = saved_model
                tokenizer_to_use = saved_tokenizer
            except Exception as e:
                print(f"✗ Error loading saved model: {str(e)}")
                print("Falling back to freshly initialized model...")
                model_to_eval = model
                tokenizer_to_use = tokenizer
        else:
            print(f"Warning: Saved model not found at {saved_model_path}")
            print("Using freshly initialized model...")
            model_to_eval = model
            tokenizer_to_use = tokenizer
        
        test_args = TrainingArguments(
            output_dir=f"{args.output_dir}/{args.model_name}",
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=args.batch_size,  
        )
        
        test_trainer = Trainer(
            model=model_to_eval,
            args=test_args,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorForSufficientFact(tokenizer_to_use, max_length=args.max_length),
        )
        
        eval_results = test_trainer.evaluate(test_dataset)
        print(eval_results)
    
if __name__ == "__main__":
    main()