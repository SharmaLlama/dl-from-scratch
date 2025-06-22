import torch
import yaml
import argparse
from datasets import load_dataset, DatasetDict
import sentencepiece as spm
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict
import random
import numpy as np
from papers.CommonTransformerComponents.train_sp import LanguageTranslationDataset, load_model
random.seed(42)
np.random.seed(42)
from utils.test_metrics_script import model_prediction, remove_trailing_periods
from sacrebleu.metrics import BLEU
import pandas as pd

def extract_long_translations(
        min_length: int,
        max_length: int,
        sp,
    ) -> Dict[str, str]:
    
    results: dict[str, str] = {}
    
    def sp_len(txt: str) -> int:
        return len(sp.encode_as_ids(txt))

    def maybe_add(en: str, hi: str) -> None:
        if  min_length <= sp_len(en) <= max_length and min_length <= sp_len(hi) <= max_length:
            results[en] = hi

    def walk_dataset(ds, nested_translation: bool) -> None:
        if isinstance(ds, DatasetDict):
            splits = ds.values()
        else:
            splits = [ds]
        for split in splits:
            for ex in split:
                if nested_translation: 
                    maybe_add(ex["translation"]["en"],
                              ex["translation"]["hi"])
                else:                                   # src / tgt style
                    maybe_add(ex["src"], ex["tgt"])

    walk_dataset(
        load_dataset("opus100", "en-hi"), nested_translation=True)

    walk_dataset(
        load_dataset("cfilt/iitb-english-hindi"), nested_translation=True)

    walk_dataset(
        load_dataset("ai4bharat/samanantar", "hi"), nested_translation=False)

    ds_hi = load_dataset("PMIndiaData/PMIndiaSum", data_dir="hindi-english")
    ds_en = load_dataset("PMIndiaData/PMIndiaSum", data_dir="english-hindi")
    en_by_url = {}
    for _, split in ds_en.items():
        for ex in split:
            en_by_url[ex["source_url"]] = ex

    for _, split in ds_hi.items():
        for ex_hi in split:
            eng_rec = en_by_url.get(ex_hi["target_url"])
            if eng_rec:
                maybe_add(eng_rec["text"], ex_hi["text"])

    return results


def sample_by_length_buckets(sentences, sp, target_samples=6000):
    """
    Sample data to limit the number of pairs in specific length buckets
    
    Args:
        sentences: list of pairs of (english, hindi) sentences
        sp: SentencePiece model 
        target_samples: Target number of samples for 0-100 and 100-200 buckets
    
    Returns:
        dictionary of sampling_info
    """
    english_encoded = sp.Encode(list(sentences.keys()))
    hindi_encoded = sp.Encode(list(sentences.values()))
    en_lengths = [len(seq) for seq in english_encoded]
    
    # Create buckets with indices
    buckets = {
        '1500_2000': [],
        '1000_1500': [],
        '0_140': [],
        '140_200': [], 
        '200_300': [],
        '300_500': [],
        '500_750': [],
        '750_1000': [],
    }
    
    for i, length in enumerate(en_lengths):
        if 0 <= length < 140:
            buckets['0_140'].append(i)
        elif 140 <= length < 200:
            buckets['140_200'].append(i)
        elif 200 <= length < 300:
            buckets['200_300'].append(i)
        elif 300 <= length < 500:
            buckets['300_500'].append(i)
        elif 500 <= length < 750:
            buckets['500_750'].append(i)
        elif 750 <= length < 1000:
            buckets['750_1000'].append(i)
        elif 1000 <= length < 1500:
            buckets['1000_1500'].append(i)
        elif 1500 <= length < 2000:
            buckets['1500_2000'].append(i)

    sampling_info = {}
    for bucket_name, indices in buckets.items():
        original_count = len(indices)
        if bucket_name in ['0_100', '100_200']:
            if original_count <= target_samples:
                sampled_indices = indices
            else:
                sampled_indices = random.sample(indices, target_samples)
        else:
            sampled_indices = indices
        
        sampled_english = [english_encoded[i] for i in sampled_indices]
        sampled_hindi = [hindi_encoded[i] for i in sampled_indices]
        sampling_info[bucket_name] = [sampled_english, sampled_hindi]
    
    return sampling_info
    
def get_bleu_score(model, dataloader, sp, device, max_seq_len=140):
    bleu = BLEU(tokenize="intl")
    full_decoded = []
    actual = []
    for batch in dataloader:
        pred = model_prediction(model, batch, max_seq_len, device, sp.bos_id(), sp.eos_id(), sp.pad_id())
        decoded = sp.decode(pred.detach().cpu().tolist())
        full_decoded.extend(remove_trailing_periods(decoded))
        actual.extend(remove_trailing_periods(sp.Decode(batch['output'].detach().cpu().tolist())))
    bs = bleu.corpus_score(full_decoded, [actual]).score
    return bs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer Long Translations")
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--llm_folder_path", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV file")    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    sp = spm.SentencePieceProcessor(model_file=args.model_file)

    sentences = extract_long_translations(50, 2000, sp)
    bucketed_sentences = sample_by_length_buckets(sentences, sp, target_samples=6000)
    result_models = {"model_name" : [], "bleu_score_0_140" : [], "bleu_score_140_200" : [],
              "bleu_score_200_300" : [], "bleu_score_300_500" : [], "bleu_score_500_750" : [],
              "bleu_score_750_1000" : [], "bleu_score_1000_1500" : [], "bleu_score_1500_2000" : []}
    
    model_number = {"sparse" : 400, "vanilla" : 250, "rope" : 350}
    model_types = ["rope", "vanilla", "sparse"]
    base_path = Path(args.llm_folder_path)
    for model_type in model_types:
        model_type_path = base_path / model_type
        for config_dir in model_type_path.iterdir():
            if config_dir.is_dir():
                model_file = config_dir / f"Model_{model_number[model_type]}"
                
                if model_file.exists():
                    model_name = f"{model_type}_{config_dir.name}_Model_{model_number[model_type]}"
                    model, config, _ = load_model(model_file, device, sp.vocab_size(), model_type=model_type, inference_len=2000)
                    model.eval()
                    for binned, pairs in bucketed_sentences.items():
                        if binned == "2000+": 
                            continue
                        max_len = int(binned.split("_")[1])
                        tmp_eng, tmp_hindi = pairs[0], pairs[1]
                        full_data = LanguageTranslationDataset(seq_length=max_len, src_encodings=tmp_eng, tgt_encodings=tmp_hindi, 
                                                            sos_token=sp.bos_id(), eos_token=sp.eos_id(), pad_token=sp.pad_id())
                        dataloader = DataLoader(full_data, batch_size=64, shuffle=True, pin_memory=True, num_workers=4)
                        try:
                            bleu_score = get_bleu_score(model, dataloader, sp, device, max_len)
                            result_models["model_name"].append(model_name)
                            result_models[f"bleu_score_{binned}"].append(bleu_score)
                            print(f"{model_name} for bucket {binned}: {bleu_score}")
                        except Exception as e:
                            result_models["model_name"].append(model_name)
                            result_models[f"bleu_score_{binned}"].append(None)
                            raise ValueError(f"Error processing {model_name} for bucket {binned}: {e}")
    df = pd.DataFrame(result_models)
    df.to_csv(f"{args.output_csv}.csv", index=False)