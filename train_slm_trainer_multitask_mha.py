import sys
import os
import torch
import logging
import argparse
import numpy as np
from torch import nn
from safetensors.torch import load_file
from torch.utils.data import Dataset
from transformers import (
    AutoConfig, AutoTokenizer, AutoModel, Trainer, TrainingArguments, PreTrainedModel
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils.config import *
from utils.utils import flatten_data
from utils.data_io import read_jsonl


logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')


class ClaimNotesDataset(Dataset):
    def __init__(self, dataset=None, tokenizer=None, max_length=0,is_flattened=False,external_evidence=False,use_summary=False):
        self.tokenizer = tokenizer
        self.max_length = max_length if max_length > 0 else 512
        self.is_flattened = is_flattened
        self.external_evidence = external_evidence
        self.use_summary = use_summary

        if not self.is_flattened:
            print("Flattening the data...")
            self.dataset=flatten_data(dataset)
        else:
            self.dataset = dataset
        for each in self.dataset:
            if 'external_link_content' not in each or each['external_link_content'] is None:
                each['external_link_content'] = {'content': '', 'summary': ''}
            else:
                if each['external_link_content'].get('content') is None:
                    each['external_link_content']['content'] = ''
                if each['external_link_content'].get('summary') is None:
                    each['external_link_content']['summary'] = ''
          
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        claim = item["claim"]
        note_text = item["note_text"]
        
        if self.external_evidence:
            external_evidence = item["external_link_content"]["content"]
            if self.use_summary:
                external_evidence_summary = item["external_link_content"]["summary"]
                text = f"Claim: {claim} [SEP] Note: {note_text} [SEP] External Link Content: {external_evidence_summary}"
            else:
                text = f"Claim: {claim} [SEP] Note: {note_text} [SEP] External Link Content: {external_evidence}"
        else:
            text = f"Claim: {claim} [SEP] Note: {note_text}"
        
        # Tokenize the text
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None
        )
        
        inputs["helpfulness_labels"] = HELPFULNESS_LABELS[item["label"]]
        reason_vec = torch.zeros(len(REASON_LABELS))
        reasons = item["reasons"].split(";")
        for reason in reasons:
            if reason in REASON_TO_IDX:
                reason_vec[REASON_TO_IDX[reason]] = 1
        inputs["reason_labels"] = reason_vec.tolist()
        
        return inputs

class DataCollatorForMultiTaskClassification:
    def __init__(self, tokenizer, padding=True, max_length=512, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        """
        The `features` parameter is a list of dictionaries, where each dictionary represents a single example
        from the dataset. Each dictionary contains the tokenized input and associated labels for that example.
        For example, each `f` in `features` might look like:
            {
                'input_ids': [...],
                'attention_mask': [...],
                'token_type_ids': [...],
                'helpfulness_labels': ...,
                'reason_labels': [...]
            }
        The collator stacks these into batched tensors for model input.
        """
        batch = {
            'input_ids': torch.tensor([f['input_ids'] for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f['attention_mask'] for f in features], dtype=torch.long),
            'token_type_ids': torch.tensor([f['token_type_ids'] for f in features], dtype=torch.long) if 'token_type_ids' in features[0] else None,
            'labels': {
                'helpfulness_labels': torch.tensor([f['helpfulness_labels'] for f in features], dtype=torch.long),
                'reason_labels': torch.tensor([f['reason_labels'] for f in features], dtype=torch.float),
            }
        }
        return batch


class MultiTaskNoteClassifier(PreTrainedModel):
    def __init__(self, config=None, label_embeddings=None):
        super().__init__(config)
        self.config = config
        self.num_reason_labels = self.config.num_reason_labels
        
        # Load pre-trained model
        self.model = AutoModel.from_pretrained(self.config._name_or_path)
        self.dropout = nn.Dropout(0.1)
        
        # Get hidden size from model config
        hidden_size = self.config.hidden_size
        
        # Label embedding integration
        self.use_label_embeddings = label_embeddings is not None
        if self.use_label_embeddings:
            print("Using label embeddings...")
            # Assuming label_embeddings is [18, 4096] tensor
            self.label_embeddings = nn.Parameter(label_embeddings.clone())
            label_embed_dim = label_embeddings.size(-1)  # 4096
            
            # Project label embeddings to match model hidden size
            self.label_projector = nn.Linear(label_embed_dim, hidden_size)
            
            # Multi-head attention for label-text interaction
            self.label_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            
            # Layer norm for attention output
            self.attention_norm = nn.LayerNorm(hidden_size)
            
            # Enhanced classifier input size (text + attended labels)
            classifier_input_size = hidden_size * 2
        else:
            classifier_input_size = hidden_size
        
        # Task-specific layers
        self.helpfulness_classifier = nn.Sequential(
            nn.Linear(classifier_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2)
        )
        
        self.reason_classifier = nn.Sequential(
            nn.Linear(classifier_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_reason_labels)
        )
        
        # Loss functions
        self.helpfulness_loss_fn = nn.CrossEntropyLoss()
        self.reason_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(list(POS_WEIGHTS.values())))
        
    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
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
        sequence_output = self.dropout(sequence_output)  # [batch_size, hidden_size]
        
        if self.use_label_embeddings:
            # Process label embeddings
            projected_labels = self.label_projector(self.label_embeddings)  # [18, hidden_size]
            
            # Prepare for attention: text as query, labels as key/value
            batch_size = sequence_output.size(0)
            text_query = sequence_output.unsqueeze(1)  # [batch_size, 1, hidden_size]
            
            # Expand labels for each batch item
            label_keys = projected_labels.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, 18, hidden_size]
            
            # Apply multi-head attention
            attended_labels, attention_weights = self.label_attention(
                query=text_query,
                key=label_keys,
                value=label_keys
            )  # attended_labels: [batch_size, 1, hidden_size]
            
            attended_labels = attended_labels.squeeze(1)  # [batch_size, hidden_size]
            attended_labels = self.attention_norm(attended_labels)
            
            # Combine text representation with attended label information
            combined_features = torch.cat([sequence_output, attended_labels], dim=-1)  # [batch_size, hidden_size*2]
        else:
            combined_features = sequence_output
        
        # Task-specific predictions
        helpfulness_logits = self.helpfulness_classifier(combined_features)
        reason_logits = self.reason_classifier(combined_features)
        
        # Calculate losses if labels are provided
        loss = None
        if labels is not None:
            helpfulness_labels = labels["helpfulness_labels"]
            reason_labels = labels["reason_labels"]
            
            if helpfulness_labels is not None and reason_labels is not None:
                helpfulness_loss = self.helpfulness_loss_fn(helpfulness_logits, helpfulness_labels)
                reason_loss = self.reason_loss_fn(reason_logits, reason_labels)
                
                # Weighted sum of losses (adjust weights as needed)
                loss = 0.4 * helpfulness_loss + 0.6 * reason_loss

        logits = {
            "helpfulness_logits": helpfulness_logits,
            "reason_logits": reason_logits,
        }
        
        # Store attention weights for analysis (optional)
        if self.use_label_embeddings and hasattr(self, 'return_attention_weights'):
            logits["attention_weights"] = attention_weights
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# Modified main function to load label embeddings
def create_model_with_label_embeddings(config, label_embeddings_path=None):
    """
    Factory function to create model with optional label embeddings
    
    Args:
        config: Model configuration
        label_embeddings_path: Path to saved label embeddings tensor file
    
    Returns:
        MultiTaskNoteClassifier instance
    """
    label_embeddings = None
    if label_embeddings_path and os.path.exists(label_embeddings_path):
        print(f"Loading label embeddings from {label_embeddings_path}")
        label_embeddings = load_file(label_embeddings_path)
        label_embeddings = label_embeddings['label_embeddings']
        print(f"Label embeddings shape: {label_embeddings.shape}")
    
    return MultiTaskNoteClassifier(config=config, label_embeddings=label_embeddings)



# --- Metrics ---
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    helpfulness_preds = predictions["helpfulness_logits"]
    reason_preds = predictions["reason_logits"]
    
    helpfulness_labels = labels["helpfulness_labels"]
    reason_labels = labels["reason_labels"]
    
    # Helpfulness metrics
    helpfulness_preds = np.argmax(helpfulness_preds, axis=1)
    helpfulness_acc = accuracy_score(helpfulness_labels, helpfulness_preds)
    helpfulness_f1 = f1_score(helpfulness_labels, helpfulness_preds, average='weighted')
    
    # Reason metrics (using threshold of 0.4)
    reason_preds_binary = (reason_preds >= 0.4).astype(int)
    precision = precision_score(reason_labels, reason_preds_binary, average='micro')
    recall = recall_score(reason_labels, reason_preds_binary, average='micro')
    micro_reason_f1 = f1_score(reason_labels, reason_preds_binary, average='micro')
    macro_reason_f1 = f1_score(reason_labels, reason_preds_binary, average='macro')
    weighted_reason_f1 = f1_score(reason_labels, reason_preds_binary, average='weighted')
    
    return {
        'helpfulness_accuracy': helpfulness_acc,
        'helpfulness_f1': helpfulness_f1,
        'precision': precision,
        'recall': recall,
        'micro_reason_f1': micro_reason_f1,
        'macro_reason_f1': macro_reason_f1,
        'weighted_reason_f1': weighted_reason_f1
    }

default_train_data_path = "./data/dataset_fixed/dataset_en_train_fixed.jsonl"
default_val_data_path = "./data/dataset_fixed/dataset_en_val_fixed.jsonl"
default_test_data_path = "./data/dataset_fixed/dataset_en_test_fixed.jsonl"
default_seed_def_path = "./data/seed_def_ordered.json"


# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=False, default=default_train_data_path)
    parser.add_argument('--val_data', type=str, required=False, default=default_val_data_path)
    parser.add_argument('--test_data', type=str, required=False, default=default_test_data_path)
    parser.add_argument('--seed_def_path', type=str, required=False, default=default_seed_def_path)
    parser.add_argument('--label_embeddings_path', type=str, default=None, help='Path to label embeddings file')
    parser.add_argument('--mode', type=str, default='train',choices=['train','eval'])
    parser.add_argument('--model_name', type=str, default='roberta-base')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='./output-multi-task')
    parser.add_argument('--save_dir', type=str, default='./save-multi-task')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--resume_from_checkpoint', type=bool, default=False)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--report_to', type=str, default='wandb',choices=['wandb','none'])
    parser.add_argument('--run_name', type=str, default='')
    args = parser.parse_args()

    # use original model or saved model
    model_name= PLM_NAME_MAPPING.get(args.model_name,args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # if the model is ModernBERT, we add an argument reference_compile=False
    config = AutoConfig.from_pretrained(model_name)
    config._name_or_path=model_name
    config.num_reason_labels=len(REASON_LABELS)
    model = create_model_with_label_embeddings(config, args.label_embeddings_path)

     # Load the datasets
    print("Loading the dataset...")
    train_data = read_jsonl(args.train_data)
    val_data = read_jsonl(args.val_data)
    train_dataset = ClaimNotesDataset(dataset=train_data, tokenizer=tokenizer,max_length=args.max_length)
    val_dataset = ClaimNotesDataset(dataset=val_data, tokenizer=tokenizer,max_length=args.max_length)
    test_data = read_jsonl(args.test_data)
    test_dataset = ClaimNotesDataset(dataset=test_data, tokenizer=tokenizer,max_length=args.max_length)

    if args.mode == 'train':
        print(f"Training the {args.model_name}...")
        training_args = TrainingArguments(
            output_dir=f"{args.output_dir}/{args.model_name}",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            # max_steps=10,
            learning_rate=args.lr,
            # save_steps=10,
            # eval_strategy="steps",
            # eval_steps=10,

            save_strategy="epoch",
            eval_strategy="epoch",
            
            logging_dir=args.log_dir,
            logging_strategy="steps",
            logging_steps=100,
            log_level="info",

            weight_decay=0.01,

            metric_for_best_model="weighted_reason_f1",
            # metric_for_best_model="eval_loss",
            # greater_is_better=True,
            save_total_limit=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            
            load_best_model_at_end=True,
            resume_from_checkpoint=args.resume_from_checkpoint,
            save_safetensors = True,
            report_to=args.report_to,
            run_name=args.run_name,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorForMultiTaskClassification(tokenizer,max_length=args.max_length),
        )
        trainer.train()
        trainer.save_model(f"{args.save_dir}/{args.model_name}")

        # test on the test data
        print(f"Evaluating the {args.model_name}...")
        eval_results=trainer.evaluate(test_dataset)
        print(eval_results)

    elif args.mode == 'eval':
        print(f"Evaluating the {args.model_name}...")
        test_args = TrainingArguments(
            do_train = False,
            do_predict = True,
            per_device_eval_batch_size = args.batch_size,  
        )
        test_trainer = Trainer(
            model=model,
            args=test_args,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorForMultiTaskClassification(tokenizer,max_length=args.max_length),
        )
        eval_results=test_trainer.evaluate(test_dataset)
        print(eval_results)
    
    
if __name__ == "__main__":
    main() 