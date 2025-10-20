#!/usr/bin/env python3
"""
Simple climate-fever fact-checking evaluation using libra_eval client.
Usage: python evaluate.py --model-name deberta-v3-base --sample 100
"""
import argparse
import json
from datetime import datetime
from utils.data_utils import load_climate_fever_data, get_available_models
from libra_eval.llmclient.local_client import Local_Client


def create_prompt(claim, evidences, helpful_evidences=None, evidence_metadata=None):
    """Create fact-checking prompt, optionally with helpfulness metadata."""
    
    # Base evidence list
    if evidence_metadata is None:
        # Simple format without helpfulness info
        evidence_text = "\n".join([f"{i+1}. {ev}" for i, ev in enumerate(evidences)])
        helpfulness_info = "Fact-check this claim using the evidence:"
    else:
        # Enhanced format with helpfulness scores and reasons
        evidence_lines = []
        for i, ev in enumerate(evidences):
            line = f"{i+1}. {ev}"
            if i < len(evidence_metadata):
                meta = evidence_metadata[i]
                helpfulness = meta.get('pred_helpfulness', 0)
                reasons = meta.get('top2_reasons', [])
                reason_text = ', '.join(reasons[:2]) if reasons else 'N/A'
                line += f"\n   [Helpfulness: {reason_text}]"
            evidence_lines.append(line)
        evidence_text = "\n".join(evidence_lines)
        helpfulness_info = "Fact-check this claim using the evidence and the helpfulness information of the evidence, if the evidence is not helpful, take less weight of the evidence."
    
    return f"""{helpfulness_info}

Claim: {claim}

Evidence:
{evidence_text}

Classify as SUPPORTS, REFUTES, NOT_ENOUGH_INFO or DISPUTED.
Format: "Classification: [YOUR_ANSWER]"
Brief reason:"""


def extract_classification(response):
    """Extract classification from model response."""
    response = response.upper()
    if "SUPPORTS" in response and "Classification:" in response:
        return "SUPPORTS"
    elif "REFUTES" in response and "Classification:" in response:
        return "REFUTES"
    elif "NOT_ENOUGH_INFO" in response and "Classification:" in response:
        return "NOT_ENOUGH_INFO"
    elif "SUPPORTS" in response:
        return "SUPPORTS"
    elif "REFUTES" in response:
        return "REFUTES"
    elif "NOT_ENOUGH" in response:
        return "NOT_ENOUGH_INFO"
    elif "DISPUTED" in response:
        return "DISPUTED"
    else:
        return "UNKNOWN"


def evaluate_climate_fever(entries, client, model="default"):
    """Evaluate climate-fever dataset with and without helpfulness using libra_eval client."""
    
    # Create base prompts (without helpfulness)
    base_messages = []
    for entry in entries:
        prompt = create_prompt(entry.claim, entry.evidences)
        base_messages.append([{"role": "user", "content": prompt}])
    
    # Create helpfulness prompts with metadata
    help_messages = []
    for entry in entries:
        # Extract evidence metadata for helpfulness prompt
        evidence_metadata = []
        for evidence_data in entry.evidences_data:  # Use the raw evidence data
            evidence_metadata.append({
                'pred_helpfulness': evidence_data.get('pred_helpfulness', 0),
                'top2_reasons': evidence_data.get('top2_reasons', [])
            })
        
        prompt = create_prompt(entry.claim, entry.evidences, entry.helpful_evidences, evidence_metadata)
        help_messages.append([{"role": "user", "content": prompt}])
    
    print(f"Running base evaluation using libra_eval client...")
    base_responses = client.multi_call(
        messages_list=base_messages,
        temperature=0.0,
        max_tokens=128
    )
    
    print(f"Running helpfulness evaluation using libra_eval client...")
    help_responses = client.multi_call(
        messages_list=help_messages,
        temperature=0.0,
        max_tokens=128
    )
    
    # Process results
    base_results = []
    help_results = []
    
    for i, entry in enumerate(entries):
        base_pred = extract_classification(base_responses[i])
        help_pred = extract_classification(help_responses[i])
        
        # Create prompts for saving
        base_prompt = create_prompt(entry.claim, entry.evidences)
        
        # Extract evidence metadata for helpfulness prompt
        evidence_metadata = []
        for evidence_data in entry.evidences_data:
            evidence_metadata.append({
                'pred_helpfulness': evidence_data.get('pred_helpfulness', 0),
                'top2_reasons': evidence_data.get('top2_reasons', [])
            })
        help_prompt = create_prompt(entry.claim, entry.evidences, entry.helpful_evidences, evidence_metadata)
        
        base_results.append({
            "claim_id": entry.claim_id,
            "claim": entry.claim,
            "evidences": entry.evidences,
            "helpful_evidences": entry.helpful_evidences,
            "prompt": base_prompt,
            "prediction": base_pred,
            "ground_truth": entry.original_claim_label,
            "correct": base_pred == entry.original_claim_label,
            "response": base_responses[i]
        })
        
        help_results.append({
            "claim_id": entry.claim_id,
            "claim": entry.claim,
            "evidences": entry.evidences,
            "helpful_evidences": entry.helpful_evidences,
            "prompt": help_prompt,
            "prediction": help_pred,
            "ground_truth": entry.original_claim_label,
            "correct": help_pred == entry.original_claim_label,
            "response": help_responses[i]
        })
    
    base_accuracy = sum(r["correct"] for r in base_results) / len(base_results)
    help_accuracy = sum(r["correct"] for r in help_results) / len(help_results)
    
    return {
        "base_results": base_results,
        "helpfulness_results": help_results,
        "base_accuracy": base_accuracy,
        "helpfulness_accuracy": help_accuracy,
        "improvement": help_accuracy - base_accuracy
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="deberta-v3-base", help="Helpfulness model name")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1", help="vLLM API base URL")
    parser.add_argument("--llm-model", default="meta-llama/Llama-3.1-8B-Instruct", help="LLM model name")
    parser.add_argument("--sample", type=int, help="Number of samples (default: all)")
    parser.add_argument("--output", default="results.json", help="Output file")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading climate-fever data for model: {args.model_name}")
    data = load_climate_fever_data(args.model_name)
    if args.sample:
        data = data[:args.sample]
    
    # Initialize libra_eval client
    api_config = {"base_url": args.api_base}
    client = Local_Client(
        model=args.llm_model,
        api_config=api_config,
        max_requests_per_minute=50  # Adjust based on your server capacity
    )
    
    # Evaluate
    results = evaluate_climate_fever(data, client, args.llm_model)
    
    # Print results
    print(f"\nResults ({len(data)} samples):")
    print(f"Base accuracy: {results['base_accuracy']:.3f}")
    print(f"Helpfulness accuracy: {results['helpfulness_accuracy']:.3f}")
    print(f"Improvement: {results['improvement']:+.3f}")
    
    # Save results
    output_data = {
        "config": {
            "model_name": args.model_name,
            "llm_model": args.llm_model,
            "sample_size": len(data),
            "api_base": args.api_base,
            "timestamp": datetime.now().isoformat()
        },
        "results": results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {args.output}")


def test():
    """Quick test without inference."""
    models = get_available_models()
    print("Available models:", models)
    
    data = load_climate_fever_data("deberta-v3-base")[:2]
    
    for entry in data:
        base_prompt = create_prompt(entry.claim, entry.evidences)
        help_prompt = create_prompt(entry.claim, entry.evidences, entry.helpful_evidences)
        
        print(f"\nClaim {entry.claim_id}: {entry.claim}")
        print(f"Evidences: {len(entry.evidences)}, Helpful: {len(entry.helpful_evidences)}")
        print(f"Ground truth: {entry.original_claim_label}")
        print(f"Base prompt length: {len(base_prompt)}")
        print(f"Help prompt length: {len(help_prompt)}")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        test()
    else:
        main()