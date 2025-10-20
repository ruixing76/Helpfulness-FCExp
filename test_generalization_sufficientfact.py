#!/usr/bin/env python3
"""
Script to test the generalization ability of the trained note classifier 
on the SufficientFact dataset by predicting evidence helpfulness towards claims.
"""

import os
import sys
import json
import argparse
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score

# Import from the unified training script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_plm_trainer_unified import (
    MultiTaskNoteClassifier,
    ClaimNotesDataset, 
    DataCollatorForMultiTaskClassification,
    batch_predict
)
from transformers import AutoTokenizer, AutoConfig
from utils.config import *
from utils.data_io import read_jsonl


def convert_sufficientfact_to_notes_format(data_path):
    """
    Convert SufficientFact dataset to the expected format for note classification
    
    Args:
        data_path: Path to SufficientFact JSONL file
        
    Returns:
        List of dictionaries in the expected format
    """
    data = read_jsonl(data_path)
    converted_data = []
    
    for item in data:
        claim = item['claim']
        evidence_list = item['evidence']
        label_after = item['label_after']
        
        # Convert evidence list to a single note text
        # Each evidence is [title, text], format as "title: text" like in train_sufficient_fact.py
        evidence_texts = []
        for evidence in evidence_list:
            if len(evidence) >= 2:
                title, text = evidence[0], evidence[1]
                evidence_texts.append(f"{title}: {text}")
        
        note_text = " ".join(evidence_texts)
        
        # Map SufficientFact labels to helpfulness (binary classification)
        # SUPPORTS/ENOUGH cases -> helpful (1)
        # NOT ENOUGH/REFUTES -> not helpful (0)
        if label_after in ["SUPPORTS", "ENOUGH -- REPEATED", "ENOUGH -- IRRELEVANT"]:
            helpfulness_label = "CURRENTLY_RATED_HELPFUL"
        else:  # NOT ENOUGH, REFUTES
            helpfulness_label = "CURRENTLY_RATED_NOT_HELPFUL"
        
        # Create a converted item
        converted_item = {
            'claim': claim,
            'note_text': note_text,
            'label': helpfulness_label,
            'reasons': "factual_accuracy",  # Default reason since we don't have ground truth
            'original_label': label_after,
            'external_link_content': {'content': '', 'summary': ''}
        }
        
        converted_data.append(converted_item)
    
    return converted_data


def get_helpfulness_from_reasons(top2_reasons, top2_probs, helpful_reasons, unhelpful_reasons):
    """
    Determine helpfulness label based on top 2 predicted reasons
    
    Args:
        top2_reasons: List of top 2 reason labels
        top2_probs: List of top 2 reason probabilities
        helpful_reasons: Set of helpful reason labels
        unhelpful_reasons: Set of unhelpful reason labels
    
    Returns:
        int: 1 for helpful, 0 for not helpful
    """
    reason_classifications = []
    
    for reason, prob in zip(top2_reasons, top2_probs):
        if reason in helpful_reasons:
            reason_classifications.append(('helpful', prob))
        elif reason in unhelpful_reasons:
            reason_classifications.append(('unhelpful', prob))
        else:
            # Unknown reason, treat as neutral/unhelpful
            reason_classifications.append(('unhelpful', prob))
    
    # Count helpful vs unhelpful reasons
    helpful_count = sum(1 for cls, _ in reason_classifications if cls == 'helpful')
    unhelpful_count = sum(1 for cls, _ in reason_classifications if cls == 'unhelpful')
    
    if helpful_count == 2:
        # Both reasons are helpful
        return 1
    elif unhelpful_count == 2:
        # Both reasons are unhelpful
        return 0
    else:
        # Mixed: one helpful, one unhelpful - use higher probability
        helpful_prob = max((prob for cls, prob in reason_classifications if cls == 'helpful'), default=0)
        unhelpful_prob = max((prob for cls, prob in reason_classifications if cls == 'unhelpful'), default=0)
        
        return 1 if helpful_prob > unhelpful_prob else 0


def analyze_predictions(predictions, original_data, output_file=None, use_reason_based_prediction=False):
    """
    Analyze the predictions and compute metrics
    
    Args:
        predictions: List of prediction dictionaries
        original_data: Original SufficientFact data
        output_file: Optional file to save detailed results
        use_reason_based_prediction: If True, derive helpfulness from top2_reasons instead of model prediction
    """
    # Define helpful and unhelpful reasons
    helpful_reasons = set(PROCESSED_HELPFUL_REASON_LABELS)
    unhelpful_reasons = set(PROCESSED_NOT_HELPFUL_REASON_LABELS)
    
    # Extract ground truth labels
    y_true = []
    y_pred = []
    y_pred_original = []  # Keep track of original predictions for comparison
    detailed_results = []
    
    for i, (pred, orig) in enumerate(zip(predictions, original_data)):
        # Ground truth mapping
        if orig['label_after'] in ["SUPPORTS", "ENOUGH -- REPEATED", "ENOUGH -- IRRELEVANT"]:
            true_label = 1  # helpful
        else:
            true_label = 0  # not helpful
        
        # Original model prediction
        original_pred_label = pred['helpfulness_label']
        
        # Reason-based prediction if requested
        if use_reason_based_prediction:
            pred_label = get_helpfulness_from_reasons(
                pred['top2_reasons'], 
                pred['top2_reason_probs'],
                helpful_reasons,
                unhelpful_reasons
            )
        else:
            pred_label = original_pred_label
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        y_pred_original.append(original_pred_label)
        
        # Detailed result for analysis
        detailed_result = {
            'index': i,
            'claim': orig['claim'],
            'evidence': orig['evidence'],
            'original_label': orig['label_after'],
            'true_helpfulness': true_label,
            'pred_helpfulness': pred_label,
            'original_pred_helpfulness': original_pred_label,
            'pred_confidence': pred['helpfulness_prob'],
            'top2_reasons': pred['top2_reasons'],
            'top2_reason_probs': pred['top2_reason_probs'],
            'correct': true_label == pred_label,
            'original_correct': true_label == original_pred_label
        }
        detailed_results.append(detailed_result)
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    precition_micro = precision_score(y_true, y_pred, average='micro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    # Generate classification report
    target_names = ['not_helpful', 'helpful']
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    print("=== Generalization Test Results ===")
    print(f"Dataset: SufficientFact")
    print(f"Total samples: {len(y_true)}")
    print(f"Prediction mode: {'Reason-based' if use_reason_based_prediction else 'Direct model prediction'}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"Precision (Micro): {precition_micro:.4f}")
    print(f"Recall (Micro): {recall_micro:.4f}")
    print(f"F1-Score (Micro): {f1_micro:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    
    # If using reason-based prediction, also show original model performance for comparison
    if use_reason_based_prediction:
        original_accuracy = accuracy_score(y_true, y_pred_original)
        original_f1_macro = f1_score(y_true, y_pred_original, average='macro')
        original_precition_micro = precision_score(y_true, y_pred_original, average='micro')
        original_recall_micro = recall_score(y_true, y_pred_original, average='micro')
        original_f1_micro = f1_score(y_true, y_pred_original, average='micro')
        original_f1_weighted = f1_score(y_true, y_pred_original, average='weighted')
        print()
        print("=== Original Model Prediction (for comparison) ===")
        print(f"Accuracy: {original_accuracy:.4f}")
        print(f"F1-Score (Macro): {original_f1_macro:.4f}")
        print(f"Precision (Micro): {original_precition_micro:.4f}")
        print(f"Recall (Micro): {original_recall_micro:.4f}")
        print(f"F1-Score (Micro): {original_f1_micro:.4f}")
        print(f"F1-Score (Weighted): {original_f1_weighted:.4f}")
    
    print()
    
    print("=== Per-Class Metrics ===")
    for class_name in target_names:
        precision = report[class_name]['precision']
        recall = report[class_name]['recall']
        f1 = report[class_name]['f1-score']
        support = report[class_name]['support']
        print(f"{class_name:12}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, Support={support}")
    print()
    
    # Analyze prediction distribution
    pred_counter = Counter(y_pred)
    true_counter = Counter(y_true)
    print("=== Label Distribution ===")
    print(f"Ground Truth: {dict(true_counter)}")
    print(f"Predictions:  {dict(pred_counter)}")
    print()
    
    # Analyze top reasons
    all_reasons = []
    for pred in predictions:
        all_reasons.extend(pred['top2_reasons'])
    
    reason_counter = Counter(all_reasons)
    print("=== Top Predicted Reasons ===")
    for reason, count in reason_counter.most_common(10):
        print(f"{reason}: {count}")
    print()
    
    # Save detailed results if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in detailed_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"Detailed results saved to {output_file}")
        
        # Also save the analysis summary to a file with the same name
        analysis_file = output_file.replace('.jsonl', '_analysis.txt') if output_file.endswith('.jsonl') else output_file + '_analysis.txt'
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write("=== Generalization Test Results ===\n")
            f.write(f"Dataset: SufficientFact\n")
            f.write(f"Total samples: {len(y_true)}\n")
            f.write(f"Prediction mode: {'Reason-based' if use_reason_based_prediction else 'Direct model prediction'}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"F1-Score (Macro): {f1_macro:.4f}\n")
            f.write(f"Precision (Micro): {precition_micro:.4f}\n")
            f.write(f"Recall (Micro): {recall_micro:.4f}\n")
            f.write(f"F1-Score (Micro): {f1_micro:.4f}\n")
            f.write(f"F1-Score (Weighted): {f1_weighted:.4f}\n")
            
            # If using reason-based prediction, also show original model performance for comparison
            if use_reason_based_prediction:
                original_accuracy = accuracy_score(y_true, y_pred_original)
                original_f1_macro = f1_score(y_true, y_pred_original, average='macro')
                original_f1_micro = f1_score(y_true, y_pred_original, average='micro')
                original_f1_weighted = f1_score(y_true, y_pred_original, average='weighted')
                f.write("\n")
                f.write("=== Original Model Prediction (for comparison) ===\n")
                f.write(f"Accuracy: {original_accuracy:.4f}\n")
                f.write(f"F1-Score (Macro): {original_f1_macro:.4f}\n")
                f.write(f"Precision (Micro): {original_precition_micro:.4f}\n")
                f.write(f"Recall (Micro): {original_recall_micro:.4f}\n")
                f.write(f"F1-Score (Micro): {original_f1_micro:.4f}\n")
                f.write(f"F1-Score (Weighted): {original_f1_weighted:.4f}\n")
            
            f.write("\n")
            f.write("=== Per-Class Metrics ===\n")
            for class_name in target_names:
                precision = report[class_name]['precision']
                recall = report[class_name]['recall']
                f1 = report[class_name]['f1-score']
                support = report[class_name]['support']
                f.write(f"{class_name:12}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, Support={support}\n")
            f.write("\n")
            
            # Analyze prediction distribution
            f.write("=== Label Distribution ===\n")
            f.write(f"Ground Truth: {dict(true_counter)}\n")
            f.write(f"Predictions:  {dict(pred_counter)}\n")
            f.write("\n")
            
            # Analyze top reasons
            f.write("=== Top Predicted Reasons ===\n")
            for reason, count in reason_counter.most_common(10):
                f.write(f"{reason}: {count}\n")
            f.write("\n")
        
        print(f"Analysis summary saved to {analysis_file}")
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'precition_micro': precition_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'classification_report': report,
        'detailed_results': detailed_results
    }


def main():
    parser = argparse.ArgumentParser(description="Test generalization on SufficientFact dataset")
    parser.add_argument('--data_path', type=str, 
                       default='./data/fc_dataset/sufficient_fact.jsonl',
                       help='Path to SufficientFact dataset')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Model name (e.g., roberta-base, deberta-v3-base)')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory containing saved models. If not provided, uses raw HuggingFace model')
    parser.add_argument('--use_mha', action='store_true',
                       help='Use multi-head attention with label embeddings')
    parser.add_argument('--label_embeddings_path', type=str, default=None,
                       help='Path to label embeddings file')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for prediction')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for prediction')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file for detailed results')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of samples for testing')
    parser.add_argument('--use_reason_based_prediction', action='store_true',
                       help='Use top2 reasons to determine helpfulness instead of direct model prediction')
    
    args = parser.parse_args()
    
    # Load and convert the dataset
    print(f"Loading SufficientFact dataset from {args.data_path}...")
    converted_data = convert_sufficientfact_to_notes_format(args.data_path)
    original_data = read_jsonl(args.data_path)
    
    if args.limit:
        converted_data = converted_data[:args.limit]
        original_data = original_data[:args.limit]
        print(f"Limited to {args.limit} samples for testing")
    
    print(f"Total samples: {len(converted_data)}")
    
    # Determine whether to use saved model or raw model
    use_saved_model = False
    model_path = args.model_name  # Default to raw model name
    
    if args.save_dir is not None:
        saved_model_path = os.path.join(args.save_dir, args.model_name)
        if os.path.exists(saved_model_path):
            model_path = saved_model_path
            use_saved_model = True
            print(f"Using trained model from: {model_path}")
        else:
            print(f"Saved model not found at {saved_model_path}, using raw model: {args.model_name}")
    else:
        print(f"No save_dir provided, using raw HuggingFace model: {args.model_name}")
    
    # Map model name if needed (for compatibility with your config)
    if not use_saved_model:
        model_path = PLM_NAME_MAPPING.get(args.model_name, args.model_name)
    
    # Load tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    config._name_or_path = model_path
    config.num_reason_labels = len(REASON_LABELS)
    
    # Create and load the model
    print(f"Creating and loading model...")
    
    # Load label embeddings if using MHA
    label_embeddings = None
    if args.use_mha and args.label_embeddings_path and os.path.exists(args.label_embeddings_path):
        print(f"Loading label embeddings from {args.label_embeddings_path}")
        from safetensors.torch import load_file
        label_embeddings = load_file(args.label_embeddings_path)
        label_embeddings = label_embeddings['label_embeddings']
        print(f"Label embeddings shape: {label_embeddings.shape}")
    
    # Create and load the model
    if use_saved_model:
        # For saved models: use from_pretrained method
        print(f"Loading trained MultiTaskNoteClassifier from {model_path}")
        model = MultiTaskNoteClassifier.from_pretrained(model_path, config=config, label_embeddings=label_embeddings, use_mha=args.use_mha)
    else:
        # For raw models: create MultiTaskNoteClassifier with base model from HuggingFace
        print("Creating MultiTaskNoteClassifier with raw base model")
        model = MultiTaskNoteClassifier(config=config, label_embeddings=label_embeddings, use_mha=args.use_mha)
    
    # Create dataset
    print("Creating dataset...")
    test_dataset = ClaimNotesDataset(
        dataset=converted_data, 
        tokenizer=tokenizer, 
        max_length=args.max_length,
        is_flattened=True
    )
    
    # Run predictions
    print("Running predictions...")
    predictions = batch_predict(
        model=model,
        dataset=test_dataset,
        data_collator=DataCollatorForMultiTaskClassification(tokenizer, max_length=args.max_length),
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Analyze results
    print("Analyzing results...")
    results = analyze_predictions(
        predictions=predictions,
        original_data=original_data,
        output_file=args.output_file,
        use_reason_based_prediction=args.use_reason_based_prediction
    )
    
    print("Generalization test completed!")


if __name__ == "__main__":
    main()