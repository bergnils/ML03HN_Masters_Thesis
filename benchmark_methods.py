import numpy as np
import nltk
import random
from summ_eval.rouge_metric import RougeMetric

def ORACLE_summarization(to_summarize, reference_summary, target_length):
    rouge_summeval = RougeMetric()
    sentences = nltk.sent_tokenize(to_summarize, language="swedish")
    summary = ""
    while (len(summary) < target_length):
        best_candidate_index = np.argmax([rouge_summeval.evaluate_example(summary=(summary + " " + s).strip(), reference=reference_summary)['rouge']['rouge_2_f_score'] for s in sentences])
        summary = summary + " " + sentences.pop(best_candidate_index)
    return summary.strip()

def RANDOM_summarization(to_summarize, target_length):
    random.seed(3030)
    sentences = nltk.sent_tokenize(to_summarize, language="swedish")
    summary = ""
    while len(summary) < target_length:
        summary = summary + " " + sentences.pop(random.choice(range(0, len(sentences))))
    return summary.strip()