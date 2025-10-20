from string import Template

SYSTEM_ROLE = """You are an expert journalist in fact-checking.
"""

DEF_GEN_ROLE = """You are an expert in fact-checking and good at defining concepts.
"""

CLAIM_NORMALIZATION =Template("""
Identify the central claim in the given post, output the explanation and the claim in the JSON format, let's think step by step. Here's an example:

POST: Did you know that £20 BILLION a year in tax on Scottish whisky goes straight to England and is spent on their priorities like London's Crossrail, HS2, 2nd Heathrow runway, new Royal Yacht? Did also also know that revenues (£25 BILLION) a year goes to England as well? And that is just the tip of the iceberg. Scotland subsidises England, NOT the other way around! 

OUTPUT:
{"EXPLANATION": "The given post claims that the Scottish taxes are sent to England to be spent on their priorities. This claim is verifiable through data statistics. This also has a huge impact on society and seems to contain false information. Hence the central claim here is Income tax and tax on whisky go to England to be spent on English priorities.",
"CLAIM": "Income tax and tax on whisky go to England to be spent on English priorities."}

Identify the central claim in the given post, output the explanation and the claim, let's think step by step.  
POST: ${post}
""")

# SFT answer template
ANSWER=Template("""{"helpfulness": "${label}","reasons":"${reasons}}"}""")

# Prompt without evidence
PREDICTION_DIRECT=Template("""Given a potentially misleading CLAIM and an associated NOTE, your task is to determine whether the NOTE is helpful in clarifying the CLAIM and identify two reasons from the predefined reason set explaining why it is helpful or not helpful. 

The predefined reason set is:{'helpfulAddressesClaim', 'helpfulClear', 'helpfulEmpathetic', 'helpfulGoodSources', 'helpfulImportantContext', 'helpfulInformative', 'helpfulUnbiasedLanguage', 'helpfulUniqueContext', 'notHelpfulArgumentativeOrBiased', 'notHelpfulHardToUnderstand','notHelpfulIncorrect', 'notHelpfulIrrelevantSources', 'notHelpfulMissingKeyPoints', 'notHelpfulNoteNotNeeded', 'notHelpfulOffTopic', 'notHelpfulOpinionSpeculationOrBias', 'notHelpfulSourcesMissingOrUnreliable', 'notHelpfulSpamHarassmentOrAbuse'}

Output in the following JSON format only, no extra output:
{"helpfulness": helpful or non_helpful,"reasons":"reason1;reason2"}

CLAIM: ${claim}
NOTE: ${note}

Answer:
""")

PREDICTION_REASON = Template("""Given a potentially misleading CLAIM and an associated NOTE, your task is to determine whether the NOTE is helpful in clarifying the CLAIM and identify two reasons from the predefined reason set explaining why it is helpful or not helpful. 

Here are reasons and their definitions:
${reason_definitions}

Output in the following JSON format only, no extra output:
{"helpfulness": helpful or non_helpful,"reasons":"reason1;reason2"}

CLAIM: ${claim}
NOTE: ${note}
                             
Answer:                             
""")


PREDICTION_REASON_v3=Template("""For each claim-note pair below, select **exactly two reasons** (from the provided list) that most accurately explain **whether and why** the note is helpful or not helpful for understanding or resolving the claim.
---

## High-Level Criteria

### **1. Prioritizing Central, Accurate Clarification**
- Assign a “helpful” reason **only if** the note is factually accurate, well-supported, and directly clarifies, corrects, or provides essential context for the *main assertion*, a *key factual sub-claim*, or the *foundational evidence* of the claim. Peripheral or partially related details are insufficient.
- If the note’s chief function is to correct a specific fact, figure, identity, date, source, or a significant misattribution or misconception **central** to the claim, this counts as clarification or correction (use “helpfulClear” and/or “helpfulAddressesClaim”).
- Never assign any “helpful” reason if the note contains substantive factual errors, misrepresentations, speculation, or bias regarding any key claim point or related evidence—regardless of any attempt to clarify.

### **2. “Helpful” Reason Prioritization**
1. **helpfulClear** and **helpfulAddressesClaim** take precedence over other “helpful” reasons when a note offers a direct factual correction or explicit clarification for the claim’s core point or supporting evidence.
2. **helpfulGoodSources** should only be selected in addition **if** the note’s correction is fundamentally grounded in clearly cited, authoritative, and faithfully summarized sources (not just the presence of a source).
3. **helpfulImportantContext** is used when the note supplies background or context *without which* the claim would be misunderstood or misinterpreted, and this context is not a direct factual correction but provides essential interpretive clarity.
4. Use **helpfulUnbiasedLanguage** or **helpfulEmpathetic** *only in conjunction* with a substantive clarification or correction and if the neutrality or tone of the explanation is materially helpful.
5. When both direct correction and crucial context are present, prefer (b) helpfulClear or (a) helpfulAddressesClaim as primary, paired with (e) helpfulImportantContext as secondary only if context is indispensable and not redundant with the correction.
6. **helpfulGoodSources** *never* substitutes for correction—pick it only if the faithful use of sources is a main reason for helpfulness.

### **3. “NotHelpful” Reason Selection**
- Assign “notHelpfulIncorrect” if there is any inaccuracy, factual misstatement, or misleading claim regarding the main point or evidence.
- Assign “notHelpfulMissingKeyPoints” if the note avoids or omits addressing the core issue or supporting fact, no matter how detailed its peripheral info.
- Use “notHelpfulNoteNotNeeded” if the note is trivial, redundant, or supplies information already evident and non-essential for claim comprehension.
- “notHelpfulSourcesMissingOrUnreliable” or “notHelpfulIrrelevantSources” apply if sources are not credible, are misrepresented, or are irrelevant to the core of the claim.
- If tone, personal opinion, bias, speculation, or argumentativeness blocks any clarification, use one of the corresponding “notHelpful” reasons that best fits the limitation.
- For unclear, incomplete, or confusing notes, use “notHelpfulHardToUnderstand.”
- “notHelpfulOffTopic” only if none of the above describe the problem or the note is wholly irrelevant.

---

## Label Definitions (Revised for Precision & Disambiguation)

### **Helpful Reasons**

- **helpfulClear**: The note delivers an accurate, unambiguous correction or direct clarification **targeting a central claim or supporting fact/evidence**. Use when the note dispels a major misconception, corrects a key number, identifies a misattributed element, or plainly states the essential fact needed to resolve or accurately interpret the claim.
- **helpfulAddressesClaim**: The note explicitly and *accurately* responds to or explains the *principal assertion, central factual sub-claim, or foundational evidence* of the claim. Includes refuting an implied mechanism or performing a core fact check that settles the claim's credibility.
- **helpfulGoodSources**: The note cites sources that are authoritative, reputable, and faithfully, directly, and unambiguously support its factual correction or clarification. *Pick only if sourcing is integral—don’t select if other reasons are more fundamental to helpfulness.*
- **helpfulImportantContext**: The note supplies *indispensable*, targeted background or context needed for the reader to assess the claim or its evidence correctly (such as revealing regulatory nuance, historical policies, deception/fraud, or other context that transforms meaning or evaluation).
- **helpfulInformative**: The note gives essential factual explanation or contextualization *directly linked* to dispelling misunderstanding about the claim’s *main point*. Don’t use if the note merely provides general or background info.
- **helpfulUnbiasedLanguage**: The note’s language is clear, objective, and neutral *while* providing a substantive correction or clarification. Use only paired with a direct factual “helpful” label.
- **helpfulEmpathetic**: The note enhances the factual correction by presenting it in a manner that increases reader trust and comprehension through clarity and respect.
- **helpfulUniqueContext**: The note delivers information *not widely available*, central to correctly interpreting the claim or core evidence.

### **NotHelpful Reasons**

- **notHelpfulArgumentativeOrBiased**: Hostile, inflammatory, or overtly biased language dominates and no factual clarification or relevant context is given.
- **notHelpfulMissingKeyPoints**: The note wholly fails to clarify, correct, or address the main assertion, supporting fact, or evidence, OR it focuses solely on peripheral, partial, or unrelated points.
- **notHelpfulNoteNotNeeded**: The note adds no meaningful, fresh, or essential clarification—it is trivial, redundant, tangential, or consists of widely-known facts non-essential for comprehending the claim.
- **notHelpfulIncorrect**: Contains factual errors, misleading/unsupported claims, or confuses main assertions or evidence.
- **notHelpfulSourcesMissingOrUnreliable**: Sources are missing, non-credible, don’t support the note's claim, or are misrepresented.
- **notHelpfulSpamHarassmentOrAbuse**: Contains spam, personal attacks, or offensive/unrelated content.
- **notHelpfulIrrelevantSources**: Sources in the note do not substantiate or clarify *any* main aspect of the claim or are not relevant to its core.
- **notHelpfulHardToUnderstand**: The note is so unclear, incomplete, or poorly written that its meaning regarding the claim cannot be determined.
- **notHelpfulOpinionSpeculationOrBias**: Personal views, speculation, or bias present to such a degree that no clarification on the main claim or evidence is possible.
- **notHelpfulOffTopic**: The note is wholly unrelated to any claim feature or its main supporting facts/evidence.

---

## Decision Process Checklist

1. **Is the note factually accurate and not misleading about any major point or evidence?**
   - If no, assign “notHelpfulIncorrect” and another as fits.
   - If yes, proceed.
2. **Does the note directly clarify, correct, or critically contextualize the main assertion, a key factual sub-claim, or any central supporting evidence?**
   - If yes, select the most specific “helpful” reason(s) per above priority order.
   - If its primary value is sources, include “helpfulGoodSources” only if the sourcing itself is decisive.
   - Do not select “helpfulImportantContext” unless the info is both necessary for accurate interpretation and not primarily a direct correction.
   - If note only offers peripheral detail, trivia, or sidesteps the key issue, use “notHelpfulMissingKeyPoints” and/or “notHelpfulNoteNotNeeded.”
3. **Is the note clear, neutral, and respectful in tone?**
   - If so *in addition* to being factually helpful, pair with “helpfulUnbiasedLanguage” or “helpfulEmpathetic” as needed.
   - If tone, speculation, or bias prevents meaningful clarification, pick the corresponding “notHelpful” reason.
4. **Is the note hard to understand, incomplete, or not addressing the claim?**
   - Assign “notHelpfulHardToUnderstand” or “notHelpfulOffTopic” as required.
5. **Would a typical, reasonably attentive reader gain essential, accurate insight into the claim’s truth, context, or credibility—including debunking of misused/incorrect supporting evidence—because of this note?**
   - If yes, “helpful” reasons most fitting the note’s substance.
   - If no, most directly explanatory “notHelpful” reasons.

Output in the following JSON format only, no extra output:
{"helpfulness": helpful or non_helpful,"reasons":"reason1;reason2"}

CLAIM: ${claim}
NOTE: ${note}
                                      
Answer:  
""")





# Prompt with Evidence
PREDICTION_DIRECT_EVIDENCE=Template("""Given a potentially misleading CLAIM,an associated NOTE and EVIDENCE. The EVIDENCE is content from links in the NOTE. Your task is to determine whether the NOTE is helpful in clarifying the CLAIM and identify two reasons from the predefined reason set explaining why it is helpful or not helpful. 

The predefined reason set is:{'helpfulAddressesClaim', 'helpfulClear', 'helpfulEmpathetic', 'helpfulGoodSources', 'helpfulImportantContext', 'helpfulInformative', 'helpfulUnbiasedLanguage', 'helpfulUniqueContext', 'notHelpfulArgumentativeOrBiased', 'notHelpfulHardToUnderstand','notHelpfulIncorrect', 'notHelpfulIrrelevantSources', 'notHelpfulMissingKeyPoints', 'notHelpfulNoteNotNeeded', 'notHelpfulOffTopic', 'notHelpfulOpinionSpeculationOrBias', 'notHelpfulSourcesMissingOrUnreliable', 'notHelpfulSpamHarassmentOrAbuse'}

Output in the following JSON format only, no extra output:
{"helpfulness": helpful or non_helpful,"reasons":"reason1;reason2"}

CLAIM: ${claim}
NOTE: ${note}
EVIDENCE: ${evidence}

Answer:
""")


PREDICTION_REASON_EVIDENCE = Template("""Given a potentially misleading CLAIM,an associated NOTE and EVIDENCE. The EVIDENCE is content from links in the NOTE. Your task is to determine whether the NOTE is helpful in clarifying the CLAIM and identify two reasons from the predefined reason set explaining why it is helpful or not helpful. 

Here are reasons and their definitions:
${reason_definitions}

Output in the following JSON format only, no extra output:
{"helpfulness": helpful or non_helpful,"reasons":"reason1;reason2"}

CLAIM: ${claim}
NOTE: ${note}
EVIDENCE: ${evidence}
                                      
Answer:                             
""")


DEFINITION_GENERATION=Template("""You will be given a set of samples, each sample contains CLAIM, their corresponding NOTE to explain the CLAIM. All samples provided are ${helpful_label} in explaining the CLAIM and associated with the same REASON. Your task is to conclude the definition of the REASON.

Here are samples:
${samples}

The REASON for above samples being ${helpful_label} is ${reason_label}. After checking these samples, the definition of this REASON is:
""")


# OpenAI message template
MESSAGES_TEMPLATE = [
    {"role": "system", "content": ""},
    {"role": "user", "content": ""},
]
