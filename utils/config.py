# PLM model name mapping
PLM_NAME_MAPPING = {
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "bert-base": "bert-base-cased",
    "bert-large": "bert-large-cased",
    "mbert": "bert-base-multilingual-cased",
    "ModernBERT-base": "answerdotai/ModernBERT-base",
    "ModernBERT-large": "answerdotai/ModernBERT-large",
    "xlm-roberta-large": "xlm-roberta-large",
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "deberta-v3-large": "microsoft/deberta-v3-large"
}

# Processed reason labels - split into helpful and not helpful categories
PROCESSED_HELPFUL_REASON_LABELS = [
    'helpfulAddressesClaim',
    'helpfulClear',
    'helpfulEmpathetic',
    'helpfulGoodSources',
    'helpfulImportantContext',
    'helpfulInformative',
    'helpfulUnbiasedLanguage',
    'helpfulUniqueContext',
]

PROCESSED_NOT_HELPFUL_REASON_LABELS = [
    'notHelpfulArgumentativeOrBiased',
    'notHelpfulHardToUnderstand',
    'notHelpfulIncorrect',
    'notHelpfulIrrelevantSources',
    'notHelpfulMissingKeyPoints',
    'notHelpfulNoteNotNeeded',
    'notHelpfulOffTopic',
    'notHelpfulOpinionSpeculationOrBias',
    'notHelpfulSourcesMissingOrUnreliable',
    'notHelpfulSpamHarassmentOrAbuse'
]


# Combined list for model training
REASON_LABELS = PROCESSED_HELPFUL_REASON_LABELS + PROCESSED_NOT_HELPFUL_REASON_LABELS

POS_WEIGHTS = {
    'helpfulAddressesClaim': 4.25,
    'helpfulClear': 3.77,
    'helpfulEmpathetic': 100,
    'helpfulGoodSources': 18.1,
    'helpfulImportantContext': 3.82,
    'helpfulInformative': 100,
    'helpfulUnbiasedLanguage': 100,
    'helpfulUniqueContext': 100,
    'notHelpfulArgumentativeOrBiased': 22.69,
    'notHelpfulHardToUnderstand': 100,
    'notHelpfulIncorrect': 42.96,
    'notHelpfulIrrelevantSources': 100,
    'notHelpfulMissingKeyPoints': 9.78,
    'notHelpfulNoteNotNeeded': 12.01,
    'notHelpfulOffTopic': 100,
    'notHelpfulOpinionSpeculationOrBias': 13.47,
    'notHelpfulSourcesMissingOrUnreliable': 56.91,
    'notHelpfulSpamHarassmentOrAbuse': 88.49
}

# Create mappings for reason labels
REASON_TO_IDX = {label: idx for idx, label in enumerate(REASON_LABELS)}
IDX_TO_REASON = {idx: label for idx, label in enumerate(REASON_LABELS)}

HELPFULNESS_LABELS={
    "CURRENTLY_RATED_HELPFUL":1,
    "CURRENTLY_RATED_NOT_HELPFUL":0
}