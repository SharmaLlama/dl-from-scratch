import pandas as pd
from pathlib import Path
import argparse
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer


def generate_reference_hindi(tokeniser, model, sentence, num):
    bos_id = tokeniser._convert_token_to_id_with_added_voc("<s>")
    eos_id = tokeniser._convert_token_to_id_with_added_voc("</s>")
    pad_id = tokeniser._convert_token_to_id_with_added_voc("<pad>")
    inp = tokeniser(f"{sentence} </s> <2hi>", add_special_tokens=False, return_tensors="pt", padding=True).input_ids 
    examples = []
    for _ in range(num):
        model_output=model.generate(inp, use_cache=False,no_repeat_ngram_size=3,encoder_no_repeat_ngram_size=3, num_beams=4, 
                                    max_length=len(inp) + 10, min_length=1, early_stopping=True, pad_token_id=pad_id, bos_token_id=bos_id, 
                                    eos_token_id=eos_id, 
                                    decoder_start_token_id=tokeniser._convert_token_to_id_with_added_voc("<2hi>"))

        decoded_output=tokeniser.decode(model_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        examples.append(decoded_output)
    
    return list(set(examples))

def get_data(datapath, skiprows, amount):
    hindi_sentences = pd.read_table(Path(datapath) /  "hindi_small.txt",  header=None, skiprows=skiprows, nrows=amount)
    return hindi_sentences.iloc[:, 0].to_list()

def write_and_generate(sentences, skiprows, tokeniser, model):
    model_dir_file = f"/srv/scratch/z3547870/en-hi/test_set_{skiprows}.txt"
    with open(model_dir_file, "w") as f:
        pass
    
    generated = []
    for idx, sentence in enumerate(sentences):
        original = [sentence]
        ref_sent = generate_reference_hindi(tokeniser, model, sentence, 5)
        original.extend(ref_sent)
        generated.append(tuple(original))

        if idx % 10 == 0:
            with open(model_dir_file, "a") as f:
                for elem in generated:
                    f.write(f"{elem}\n")
            generated = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Case Ref Generation")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--skiprows", type=int, required=True)
    parser.add_argument("--amount", type=int, required=True)
    args = parser.parse_args()

    tokeniser = AutoTokenizer.from_pretrained("ai4bharat/MultiIndicParaphraseGeneration", do_lower_case=False, use_fast=False, keep_accents=True)
    model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/MultiIndicParaphraseGeneration")

    hindi_sentences = get_data(args.dataset, args.skiprows, args.amount)
    write_and_generate(hindi_sentences, args.skiprows, tokeniser, model)
