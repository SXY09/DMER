CDR_TEMPLATE_REACT_EN = \
"""Your task is to CORRECT the wrong relation triples (pred_spo) to match the ground truth (golden_spo). 
You MUST NOT extract new triples, delete any existing entity pairs in pred_spo, or swap subject/object—only fix the 'predicate' of each existing entity pair.


### Key Rules for No Ground Truth (golden_spo = null):
- You CANNOT rely on golden_spo. Instead, you must use TWO tools to verify:
  1. `RetrieveCorrectMemory`: Get correct labeling examples of "chemical-disease" relations to align standards.
  2. `RetrieveReflexionMemory`: Get common wrong cases to avoid repeating errors.
- Final predicate must be one of: `chemical induced disease|no chemical disease induction relation` (no other options).


You can use the following tools:
{tools}

### Correction Steps (MUST follow in order):
1. **Compare Errors**: List all triples in pred_spo and golden_spo, mark which 'predicate' fields are different.
2. **Verify Relations**: Use GetRelationDefinition to confirm the meaning of predicates, and RetrieveCorrectMemory to find similar correct examples.
3. **Fix Predicates**: Only modify the 'predicate' in pred_spo to match golden_spo (keep subjects/objects unchanged).
4. **Check Completeness**: Ensure your output has EXACTLY the same number of triples as golden_spo (no missing/extra).


### Output Format (Strictly follow):
Thought: Explain your analysis of errors (which triples are wrong, why)
Action: Tool name (from the list above)
ActionInput: Parameter for the tool (e.g., text for retrieval, relation names for definition)
Observation: Result returned by the tool
... (Repeat until you call Finish with corrected triples)

Begin!
The input sentence is `{text}`\n
"""
CDR_FIRST_STEP_EN = \
"""Thought: First, I need to know more about the definition and output format of the relation triple extraction task.
Action: GetTaskDescription
ActionInput:
Observation: {task_description}\n"""
CDR_SECOND_STEP_EN = \
"""Thought: I can first observe some already labeled relation triples to better understand this task.
Action: RetrieveExamples
ActionInput: {text}
Observation: {retrieved_examples}\n"""

CDR_SUFFIX = """Thought: Now I will compare pred_spo and golden_spo to find wrong predicates:
- pred_spo: {pred_spo}
- golden_spo: {golden_spo}
First, I'll list the triples with mismatched predicates...
Current Check: Have you met any termination condition (e.g., corrected all predicates, got enough examples)? If yes, call "Finish" immediately.
"""

CDR_SECOND_STEP_MEMORY_EN = \
"""Thought: I can find some examples from the existing correct examples to help me understand this task.
Action: RetrieveCorrectMemory
ActionInput: {text}
Observation: {retrieved_examples}\n"""

CDR_THIRD_STEP_REFLEXION_EN = \
"""Thought: I need common wrong cases to avoid mistakes .
Action: RetrieveReflexionMemory
ActionInput: {text}
Observation: {retrieved_reflexion_examples}\n"""

CDR_FOURTH_STEP_ANALYZE_EN = \
"""Now I will analyze pred_spo comprehensively to correct wrong predicates, based on text evidence, correct cases from RetrieveCorrectMemory, and mistake warnings from RetrieveReflexionMemory:
- pred_spo: {pred_spo}
keep all pred_spo entity pairs
Current Check: Have you met any termination condition (e.g., corrected all predicates, got enough examples)? If yes, call "Finish" immediately.
"""

CDR_TEMPLATE_REFLEXION_EN = \
"""In the relation extraction task, for the input sentence `{text}`, the correct result should be `{golden}`. But the model's output result is `{pred}`.
Please summarize the reason for the error in one sentence: """

CDR_TEMPLATE_SUMMAY_EN = \
"""In the relation extraction task, for the input sentence `{text}`, the correct result should be `{golden}`. Here is the extraction process that can be referred to:
```
{history}
```
If you cannot perform these actions in the extraction process and need to directly generate the extraction result, please give your reasoning in one sentence and give the final JSON extraction result: """



GDA_TEMPLATE_REACT_EN = \
"""Your task is to CORRECT the wrong relation triples (pred_spo) to match the ground truth (golden_spo). 
You MUST NOT extract new triples, delete any existing entity pairs in pred_spo, or swap subject/object—only fix the 'predicate' of each existing entity pair.


### Key Rules for No Ground Truth (golden_spo = null):
- You CANNOT rely on golden_spo. Instead, you must use TWO tools to verify:
  1. `RetrieveCorrectMemory`: Get correct labeling examples of "gene-disease association" relations to align standards.
  2. `RetrieveReflexionMemory`: Get common wrong cases to avoid repeating errors.
- Final predicate must be one of: `gene disease association|no gene disease association` (no other options).


You can use the following tools:
{tools}

### Correction Steps (MUST follow in order):
1. **Compare Errors**: List all triples in pred_spo and golden_spo, mark which 'predicate' fields are different.
2. **Verify Relations**: Use GetRelationDefinition to confirm the meaning of predicates, and RetrieveCorrectMemory to find similar correct examples.
3. **Fix Predicates**: Only modify the 'predicate' in pred_spo to match golden_spo (keep subjects/objects unchanged).
4. **Check Completeness**: Ensure your output has EXACTLY the same number of triples as golden_spo (no missing/extra).


### Output Format (Strictly follow):
Thought: Explain your analysis of errors (which triples are wrong, why)
Action: Tool name (from the list above)
ActionInput: Parameter for the tool (e.g., text for retrieval, relation names for definition)
Observation: Result returned by the tool
... (Repeat until you call Finish with corrected triples)

Begin!
The input sentence is `{text}`\n
"""
GDA_FIRST_STEP_EN = \
"""Thought: First, I need to know more about the definition and output format of the relation triple extraction task.
Action: GetTaskDescription
ActionInput:
Observation: {task_description}\n"""
GDA_SECOND_STEP_EN = \
"""Thought: I can first observe some already labeled relation triples to better understand this task.
Action: RetrieveExamples
ActionInput: {text}
Observation: {retrieved_examples}\n"""

GDA_SUFFIX = """Thought: Now I will compare pred_spo and golden_spo to find wrong predicates:
- pred_spo: {pred_spo}
- golden_spo: {golden_spo}
First, I'll list the triples with mismatched predicates...
Current Check: Have you met any termination condition (e.g., corrected all predicates, got enough examples)? If yes, call "Finish" immediately.
"""

GDA_SECOND_STEP_MEMORY_EN = \
"""Thought: I can find some examples from the existing correct examples to help me understand this task.
Action: RetrieveCorrectMemory
ActionInput: {text}
Observation: {retrieved_examples}\n"""

GDA_THIRD_STEP_REFLEXION_EN = \
"""Thought: I need common wrong cases to avoid mistakes .
Action: RetrieveReflexionMemory
ActionInput: {text}
Observation: {retrieved_reflexion_examples}\n"""

GDA_FOURTH_STEP_ANALYZE_EN = \
"""Now I will analyze pred_spo comprehensively to correct wrong predicates, based on text evidence, correct cases from RetrieveCorrectMemory, and mistake warnings from RetrieveReflexionMemory:
- pred_spo: {pred_spo}
keep all pred_spo entity pairs
Current Check: Have you met any termination condition (e.g., corrected all predicates, got enough examples)? If yes, call "Finish" immediately.
"""

GDA_TEMPLATE_REFLEXION_EN = \
"""In the relation extraction task, for the input sentence `{text}`, the correct result should be `{golden}`. But the model's output result is `{pred}`.
Please summarize the reason for the error in one sentence: """

GDA_TEMPLATE_SUMMAY_EN = \
"""In the relation extraction task, for the input sentence `{text}`, the correct result should be `{golden}`. Here is the extraction process that can be referred to:
```
{history}
```
If you cannot perform these actions in the extraction process and need to directly generate the extraction result, please give your reasoning in one sentence and give the final JSON extraction result: """


CHR_TEMPLATE_REACT_EN = \
"""Your task is to CORRECT the wrong relation triples (pred_spo) to match the ground truth (golden_spo). 
You MUST NOT extract new triples, delete any existing entity pairs in pred_spo, or swap subject/object—only fix the 'predicate' of each existing entity pair.


### Key Rules for No Ground Truth (golden_spo = null):
- You CANNOT rely on golden_spo. Instead, you must use TWO tools to verify:
  1. `RetrieveCorrectMemory`: Get correct labeling examples of "chemical-metabolite" relations to align standards.
  2. `RetrieveReflexionMemory`: Get common wrong cases to avoid repeating errors.
- Final predicate must be one of: `chemical metabolite interaction|no chemical metabolite interaction` (no other options).


You can use the following tools:
{tools}

### Correction Steps (MUST follow in order):
1. **Compare Errors**: List all triples in pred_spo and golden_spo, mark which 'predicate' fields are different.
2. **Verify Relations**: Use GetRelationDefinition to confirm the meaning of predicates, and RetrieveCorrectMemory to find similar correct examples (e.g., how "metabolic pathway" or "substrate conversion" indicates interaction).
3. **Fix Predicates**: Only modify the 'predicate' in pred_spo to match golden_spo (keep subjects/objects unchanged).
4. **Check Completeness**: Ensure your output has EXACTLY the same number of triples as golden_spo (no missing/extra).


### Output Format (Strictly follow):
Thought: Explain your analysis of errors (which triples are wrong, why—e.g., "Triple 1: predicate should be 'chemical metabolite interaction' because text mentions 'glucose is converted to pyruvate'")
Action: Tool name (from the list above)
ActionInput: Parameter for the tool (e.g., "glucose and pyruvate" for retrieval, "chemical metabolite interaction" for definition)
Observation: Result returned by the tool
... (Repeat until you call Finish with corrected triples)

Begin!
The input sentence is `{text}`\n
"""

CHR_FIRST_STEP_EN = \
"""Thought: First, I need to know more about the definition and output format of chemical-metabolite interaction extraction.
Action: GetTaskDescription
ActionInput:
Observation: {task_description}\n"""

CHR_SECOND_STEP_EN = \
"""Thought: I can first observe some labeled chemical-metabolite relation triples to understand how to judge interactions (e.g., based on metabolic keywords like 'synthesis' or 'catalysis').
Action: RetrieveExamples
ActionInput: {text}
Observation: {retrieved_examples}\n"""

CHR_SUFFIX = """Thought: Now I will compare pred_spo and golden_spo to find wrong predicates:
- pred_spo: {pred_spo}
- golden_spo: {golden_spo}
First, I'll list the triples with mismatched predicates and check text evidence (e.g., 'metabolic pathway' supports interaction, 'no statistical correlation' supports no interaction)...
Current Check: Have you met any termination condition (e.g., corrected all predicates, got enough examples)? If yes, call "Finish" immediately.
"""

CHR_SECOND_STEP_MEMORY_EN = \
"""Thought: I can find correct examples of chemical-metabolite relations from existing memory to learn how to label interactions (e.g., when to use 'no interaction' for unrelated chemicals).
Action: RetrieveCorrectMemory
ActionInput: {text}
Observation: {retrieved_examples}\n"""

CHR_THIRD_STEP_REFLEXION_EN = \
"""Thought: I need to check common mistakes in chemical-metabolite extraction (e.g., misjudging 'substrate' as no interaction) to avoid repeating them.
Action: RetrieveReflexionMemory
ActionInput: {text}
Observation: {retrieved_reflexion_examples}\n"""

CHR_FOURTH_STEP_ANALYZE_EN = \
"""Now I will analyze pred_spo comprehensively to correct wrong predicates, based on:
- Text evidence (e.g., 'catalyses' indicates interaction, 'unrelated metabolic pathway' indicates no interaction)
- Correct cases from RetrieveCorrectMemory
- Mistake warnings from RetrieveReflexionMemory
- pred_spo: {pred_spo}
Keep all entity pairs in pred_spo unchanged.
Current Check: Have you met any termination condition (e.g., corrected all predicates, got enough examples)? If yes, call "Finish" immediately.
"""

CHR_TEMPLATE_REFLEXION_EN = \
"""In the chemical-metabolite interaction extraction task, for the input sentence `{text}`, the correct result should be `{golden}`. But the model's output result is `{pred}`.
Please summarize the error reason in one sentence (e.g., "Missed 'metabolic conversion' in text, leading to wrong 'no interaction' label"): """

CHR_TEMPLATE_SUMMAY_EN = \
"""In the relation extraction task, for the input sentence `{text}`, the correct result should be `{golden}`. Here is the extraction process that can be referred to:
```
{history}
```
If you cannot perform these actions in the extraction process and need to directly generate the extraction result, please give your reasoning in one sentence and give the final JSON extraction result: """

SUFFIX = """Thought: """