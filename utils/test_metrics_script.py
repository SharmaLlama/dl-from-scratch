from pathlib import Path
import sentencepiece as spm
import torch
import numpy as np
import random
import torch.nn.functional as F
import argparse
from sacrebleu.metrics import BLEU
from papers.CommonTransformerComponents.train_sp import load_model, get_dataloaders, get_encodings

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
def remove_trailing_periods(sentences):
    return [s[:-1] if s.endswith('.') else s for s in sentences]

def model_prediction(model, batch, max_len, device, sos_token, eos_token, pad_token):
    with torch.inference_mode():

        underlying_model = model.module if hasattr(model, 'module') else model  

        encoder_input = batch['src'].to(device)  # B x seq_len
        encoder_mask = batch['encoder_mask'].to(device)  # B x 1 x 1 x seq_len
        encoder_output = underlying_model.encode(encoder_input, encoder_mask)
        torch.cuda.empty_cache()
        B = encoder_input.size(0)
        decoder_input = torch.full((B, max_len), pad_token).to(device)
        decoder_input[:, 0] = sos_token
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(max_len - 1):
            subsequent_mask = torch.tril(torch.ones((max_len, max_len), dtype=torch.int)).expand(B, -1, -1).to(device)
            other_mask = (decoder_input != pad_token).int().unsqueeze(1)
            decoder_mask = (subsequent_mask & other_mask).unsqueeze(1).to(device)
            out = underlying_model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
            prediction = underlying_model.proj(out)  # shape: (B, max_len, vocab_size)
            next_tokens = torch.argmax(prediction[:, t, :], dim=-1)
            next_tokens = torch.where(finished, pad_token, next_tokens)
            decoder_input[:, t + 1] = next_tokens
            finished |= (next_tokens == eos_token)
            if finished.all():
                break

    return decoder_input

def get_bleu_score(model, dataloader, sp, device, config):
    bleu = BLEU(tokenize="intl")
    full_decoded = []
    actual = []
    for batch in dataloader:
        pred = model_prediction(model, batch, config['SEQ_LEN'], device, sp.bos_id(), sp.eos_id(), sp.pad_id())
        decoded = sp.decode(pred.detach().cpu().tolist())
        full_decoded.extend(remove_trailing_periods(decoded))
        actual.extend(remove_trailing_periods(sp.Decode(batch['output'].detach().cpu().tolist())))
    bs = bleu.corpus_score(full_decoded, [actual]).score
    return bs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer Testing Script")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--amount", type=int, required=True)
    parser.add_argument("--llm_folder_path", type=str, required=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    bleu = BLEU(tokenize="intl")
    sp = spm.SentencePieceProcessor(model_file=args.model_file)
    english_encoded, hindi_encoded, ref_sentences = get_encodings(args.dataset, skiprows=550_000, amount=args.amount, sp=sp)
    dataloader = get_dataloaders(sp, english_encoded, hindi_encoded, 512) # fixed batch size here

    model_number = {"sparse" : 400, "vanilla" : 250, "rope" : 350}
    model_types = ["sparse"] #, "vanilla", "rope"]
    base_path = Path(args.llm_folder_path)
    for model_type in model_types:
        model_type_path = base_path / model_type
        for config_dir in model_type_path.iterdir():
            if config_dir.is_dir():
                model_file = config_dir / f"Model_{model_number[model_type]}"
                if model_file.exists():
                    model_name = f"{model_type}_{config_dir.name}_Model_{model_number[model_type]}"
                    model, config, _ = load_model(model_file, device, model_type=model_type)
                    model.eval()
                    bleu_score = get_bleu_score(model, dataloader, sp, device, config)
                    print(f"Model: {config_dir.name}, BLEU Score: {bleu_score:.4f}")