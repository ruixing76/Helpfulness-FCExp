# Transform data to have one entry per claim-note pair
def flatten_data(data):
    flattened_data=[]
    for item in data:
        claim = item["claim"]
        for note in item["notes"]:
            flattened_data.append({
                "claim": claim,
                "note_text": note["text"],
                "reasons": note.get("reasons", ""),
                "label": note["label"]  # Use the label associated with the note
            })
    return flattened_data