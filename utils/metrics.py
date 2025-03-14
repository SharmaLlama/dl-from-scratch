from indicnlp.tokenize import indic_tokenize
from collections import Counter

from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
import numpy as np

def generate_reference_hindi(sentence, num):
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/MultiIndicParaphraseGeneration", do_lower_case=False, use_fast=False, keep_accents=True)
    model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/MultiIndicParaphraseGeneration")

    bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
    eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
    pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")
    inp = tokenizer(f"{sentence} </s> <2hi>", add_special_tokens=False, return_tensors="pt", padding=True).input_ids 
    examples = []
    for _ in range(num):
        model_output=model.generate(inp, use_cache=False,no_repeat_ngram_size=3,encoder_no_repeat_ngram_size=3, num_beams=4, 
                                    max_length=len(inp) + 10, min_length=1, early_stopping=True, pad_token_id=pad_id, bos_token_id=bos_id, 
                                    eos_token_id=eos_id, 
                                    decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc("<2hi>"))

        decoded_output=tokenizer.decode(model_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        examples.append(decoded_output)
    
    return list(set(examples))


def brevity_penality(candidate_len, ref_len):
    if candidate_len > ref_len:
        return 1
    else:
        return np.exp(1 - ref_len/candidate_len)

def clipped_precision(references, candidate, n):
    n_gram_candidate = [tuple(candidate[i: i + n]) for i in range(len(candidate) - n + 1)]
    candidate_counts = Counter(n_gram_candidate)
    n_gram_references  = [
        [tuple(ref[i:i + n]) for i in range(len(ref) - n + 1)] for ref in references
    ]

    merged_max_counts = Counter()
    for n_gram_ref in n_gram_references:
        reference_counts = Counter(n_gram_ref)
        for k,v in reference_counts.items():
            merged_max_counts[k] = max(merged_max_counts[k], v)

    clipped_counts = {ngram : min(merged_max_counts[ngram], count) for ngram, count in candidate_counts.items()} 
    total_clipped = sum(clipped_counts.values())
    total_generated = sum(candidate_counts.values())

    return total_clipped, total_generated

def sentence_bleu(reference : list, candidate : str, weights=(0.25, 0.25, 0.25, 0.25), generate_more=False):
    if generate_more:
        reference += generate_reference_hindi(reference[0], num=3)
    print(f"generated {len(reference) - 1} more sentences")
    candidate_token = indic_tokenize.trivial_tokenize(candidate)
    reference_tokens = [indic_tokenize.trivial_tokenize(ref) for ref in reference]
    precisions = [clipped_precision(reference_tokens, candidate_token, n) for n in range(1, 5)]
    log_sum = sum(w * np.log(p[0] / p[1]) if p[1] > 0 and p[0] > 0 else 0 for w,p in zip(weights, precisions))
    final_score = brevity_penality(len(candidate), closest_ref_len(reference, len(candidate))) * np.exp(log_sum)
    return final_score

def closest_ref_len(references, candidate_len):
    ref_lens = (len(ref) for ref in references)
    closest_len = min(ref_lens, 
                      key = lambda ref_len: (abs(ref_len - candidate_len), ref_len)
                      )
    return closest_len

def corpus_bleu(references: list[list], candidates : list[str], weights=(0.25, 0.25, 0.25, 0.25)):
    numerator = Counter()
    denominator = Counter()
    candidate_len, ref_len = 0, 0
    for reference_list, candidate in zip(references, candidates):
        candidate_token = indic_tokenize.trivial_tokenize(candidate)
        reference_tokens = [indic_tokenize.trivial_tokenize(ref) for ref in reference_list]

        for i in range(1,5):
            p_num, p_denom = clipped_precision(reference_tokens, candidate_token, i)
            numerator[i] += p_num
            denominator[i] += p_denom
        
        cand_len = len(candidate)
        candidate_len += cand_len
        ref_len += closest_ref_len(references, cand_len)


    bp = brevity_penality(candidate_len, ref_len)
    pn = [numerator[i] / denominator[i] if denominator[i] > 0 else 0 for i in range(1, 5)]
    log_sum = sum([w * np.log(p) if p > 0 else 0 for w,p in zip(weights, pn)])
    return bp * np.exp(log_sum)

