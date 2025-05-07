
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
import argparse
import yaml
import sentencepiece as spm
import numpy as np
from utils.metrics import corpus_bleu
import numpy as np
from papers.attention_is_all_you_need.TransformerComponents.Encoder import Encoder
from papers.attention_is_all_you_need.TransformerComponents.Decoder import Decoder
from papers.attention_is_all_you_need.TransformerComponents.PE import PositionalEmbedding
from papers.attention_is_all_you_need.TransformerComponents.Transformer import Transformer
from papers.attention_is_all_you_need.TransformerComponents.UtilsLayers import Projection
from sacrebleu.metrics import BLEU

import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# YAML_PATH = "dl-from-scratch/papers/attention_is_all_you_need/config.yaml"
YAML_PATH = Path("dl-from-scratch")  / "papers" / "attention_is_all_you_need" / "config.yaml"
with open(YAML_PATH, "r") as file:
    config = yaml.safe_load(file)

class LanguageTranslationDataset(Dataset):
    def __init__(self, seq_length, src_encodings, tgt_encodings, sos_token, eos_token, pad_token):
        super().__init__()
        self.paired_encodings = LanguageTranslationDataset.augment_encodings(src_encodings, tgt_encodings, sos_token, eos_token)
        self.seq_len = seq_length
        self.pad_token = pad_token

    @staticmethod
    def augment_encodings(src_encodings, tgt_encodings, sos_token, eos_token):
        src_encodings_app = [[sos_token] + sublist + [eos_token] for sublist in src_encodings]
        tgt_encodings_app = [[sos_token] + sublist for sublist in tgt_encodings]
        output_encodings = [sublist + [eos_token] for sublist in tgt_encodings] 
        full_encoding = list(zip(src_encodings_app, tgt_encodings_app, output_encodings))
        full_encoding.sort(key=lambda x: len(x[0])) # sort sequence lengths
        return full_encoding


    def __getitem__(self, idx):
            src_seq, tgt_seq, output_seq = self.paired_encodings[idx]
            
            # Convert to tensors
            src_tensor = torch.tensor(src_seq, dtype=torch.long)
            tgt_tensor = torch.tensor(tgt_seq, dtype=torch.long)
            output_tensor = torch.tensor(output_seq, dtype=torch.long)

            # **Ensure the sequence length does not exceed `seq_len`**
            if src_tensor.size(0) > self.seq_len:
                src_tensor = src_tensor[:self.seq_len]  # Crop excess tokens
            if tgt_tensor.size(0) > self.seq_len:
                tgt_tensor = tgt_tensor[:self.seq_len]  # Crop excess tokens
            if output_tensor.size(0) > self.seq_len:
                output_tensor = output_tensor[:self.seq_len]  # Crop excess tokens

            src_tensor = F.pad(src_tensor, (0, max(0, self.seq_len - src_tensor.size(0))), value=self.pad_token)
            tgt_tensor = F.pad(tgt_tensor, (0, max(0, self.seq_len - tgt_tensor.size(0))), value=self.pad_token)
            output_tensor = F.pad(output_tensor, (0, max(0, self.seq_len - output_tensor.size(0))), value=self.pad_token)
            encoder_mask = (src_tensor != self.pad_token).int()
            subsequent_mask = torch.tril(torch.ones((self.seq_len, self.seq_len), dtype=torch.int))
            padding_mask = (tgt_tensor != self.pad_token).int()
            decoder_mask = subsequent_mask & padding_mask.unsqueeze(0)



            return {
                "src": src_tensor, # Seq_len
                "tgt": tgt_tensor, # seq_len
                "output": output_tensor, # seq_len
                "encoder_mask" : encoder_mask.unsqueeze(0).unsqueeze(0), # 1 x 1 x seq_len
                "decoder_mask" : decoder_mask.unsqueeze(0), # 1 x seq_len x seq_len
            }
    def __len__(self): 
            return len(self.paired_encodings)
    
def build_model(sp, device, state_dict=None):
    vocab_size = sp.vocab_size()
    encoder_transformer = Encoder(config['N_ENCODERS'], config['N_HEADS'], config['D_MODEL'], config['D_MODEL'] // config['N_HEADS'], config['D_MODEL'] // config['N_HEADS'], config['FF_HIDDEN'], config['DROPOUT'])
    decoder_transformer = Decoder(config['N_DECODERS'], config['N_HEADS'], config['D_MODEL'], config['D_MODEL'] // config['N_HEADS'], config['D_MODEL'] // config['N_HEADS'], config['FF_HIDDEN'], config['DROPOUT'])
    src_embeddings = PositionalEmbedding(vocab_size, config['D_MODEL'], config['SEQ_LEN'], config['DROPOUT'])
    tgt_embeddings = PositionalEmbedding(vocab_size, config['D_MODEL'], config['SEQ_LEN'], config['DROPOUT'])
    projection = Projection(config['D_MODEL'], vocab_size)
    model = Transformer(encoder_transformer, decoder_transformer, src_embeddings, tgt_embeddings, projection).to(device)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    if state_dict is not None:
        if torch.cuda.device_count() > 1:
            model.load_state_dict(state_dict)
        else: 
            state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
            model.load_state_dict(state_dict)
    elif torch.cuda.device_count() > 1:
        model.module.initialise()
    else: 
        model.initialise() 
    return model

def get_encodings(datapath, skiprows, amount, sp):
    english_sentences = pd.read_table(Path(datapath) /  "english_small.txt",  header=None, skiprows=skiprows, nrows=100_000)
    hindi_sentences = pd.read_table(Path(datapath) /  "hindi_small.txt",  header=None, skiprows=skiprows, nrows=100_000)
    idxs = random.sample(range(100_000), amount)
    english_encoded = sp.encode_as_ids(english_sentences.iloc[idxs, 0].to_list())
    hindi_encoded = sp.encode_as_ids(hindi_sentences.iloc[idxs, 0].to_list())
    return english_encoded, hindi_encoded

def get_dataloaders(sp, english_encoded, tgt_encoded, amount):
    full_data = LanguageTranslationDataset(seq_length=config['SEQ_LEN'], src_encodings=english_encoded, tgt_encodings=tgt_encoded, sos_token=sp.bos_id(), eos_token=sp.eos_id(),
                                        pad_token=sp.pad_id())
    dataloader = DataLoader(full_data, batch_size=amount, pin_memory=False, shuffle=False)
    return dataloader


def create_masks(ablation_mask, default_value, device):
    if ablation_mask is not None:
        ablation_masks = []
        for layer in ablation_mask:
            temp_mask = default_value.to(device).repeat(1, len(layer), 1, 1)
            for idx, h in enumerate(layer):
                temp_mask[:, idx, ...] *= h
            ablation_masks.append(temp_mask)
    else:
        ablation_masks = default_value.to(device)
    return ablation_masks

def model_prediction_ablation(model, batch, max_len, device, sp, encoder_heads=None, decoder_heads=None, encoder_decoder_heads=None):
    # each of the heads argument will be a list of lists is specified of 1s and 0s where 1 means that head is not abalated and 0 means it is.
    # The length of the outer list should be equal to number of layers and each inner list should be number of heads.    
    with torch.inference_mode():
        encoder_masks = create_masks(encoder_heads, batch['encoder_mask'], device)
        encoder_decoder_masks = create_masks(encoder_decoder_heads, batch['encoder_mask'], device) if encoder_decoder_heads is not None else None
        underlying_model = model.module if hasattr(model, 'module') else model
        encoder_input = batch['src'].to(device)  # B x seq_len
        encoder_output = underlying_model.encode(encoder_input, encoder_masks)
        B = encoder_input.size(0)
        decoder_input = torch.full((B, max_len), sp.pad_id()).to(device)
        decoder_input[:, 0] = sp.bos_id()
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        subsequent_mask = torch.tril(torch.ones((max_len, max_len), dtype=torch.int)).repeat(B, 1, 1).to(device)
        for t in range(max_len - 1):
            other_mask = (decoder_input != sp.pad_id()).int().unsqueeze(1)
            decoder_mask = (subsequent_mask & other_mask).unsqueeze(1)
            decoder_mask = create_masks(decoder_heads, decoder_mask, device)
            if isinstance(encoder_decoder_masks, list):
                out = underlying_model.decode(decoder_input, encoder_output, encoder_mask=encoder_decoder_masks, decoder_mask=decoder_mask)
            else:
                out = underlying_model.decode(decoder_input, encoder_output, encoder_mask=encoder_masks, decoder_mask=decoder_mask)
            
            prediction = underlying_model.proj(out)  # shape: (B, max_len, vocab_size)
            next_tokens = torch.argmax(prediction[:, t, :], dim=-1)
            next_tokens = torch.where(finished, sp.pad_id(), next_tokens)
            decoder_input[:, t + 1] = next_tokens
            finished |= (next_tokens == sp.eos_id())
            if finished.all():
                break
        return decoder_input

def remove_trailing_periods(sentences):
    return [s[:-1] if s.endswith('.') else s for s in sentences]

def ablation_studies(model, dataloader, device, sp, heads=4, n_encoders=2, n_decoders=2):
    removal_order = []
    bleu_scores_removal = []
    last_bleu_score = 100
    total_layers = n_encoders + 2 * n_decoders
    bleu = BLEU(tokenize='intl')

    translated = []
    actual = []
    for batch in dataloader:
        pred = model_prediction_ablation(
            model, 
            batch, 
            140, 
            device, 
            sp, 
            encoder_heads=None, 
            decoder_heads=None, 
            encoder_decoder_heads=None
        )
        
        translated.extend(remove_trailing_periods(sp.Decode(pred.detach().cpu().tolist())))
        actual.extend(remove_trailing_periods(sp.Decode(batch['output'].detach().cpu().tolist())))

    print(f"baseline_bleu sb: {bleu.corpus_bleu(translated, [actual]).score}")
    print(heads * total_layers)
    while len(removal_order) < (heads * total_layers) and last_bleu_score > 5:
        bleu_scores = {}
        
        for remover in range(heads * total_layers):
            if remover in removal_order:
                continue
                
            divider, rem = divmod(remover, heads)
            encoder_heads = [[1 for _ in range(heads)] for _ in range(n_encoders)]
            decoder_heads = [[1 for _ in range(heads)] for _ in range(n_decoders)]
            encoder_decoder_heads = [[1 for _ in range(heads)] for _ in range(n_decoders)]
            
            for prev_removal in removal_order:
                prev_div, prev_rem = divmod(prev_removal, heads)
                if prev_div < n_encoders:
                    encoder_heads[prev_div][prev_rem] = 0
                elif prev_div < (n_encoders + n_decoders):
                    decoder_heads[prev_div-n_encoders][prev_rem] = 0
                else:
                    encoder_decoder_heads[prev_div-(n_encoders + n_decoders)][prev_rem] = 0
            
            if divider < n_encoders:
                encoder_heads[divider][rem] = 0
            elif divider < (n_encoders + n_decoders):
                decoder_heads[divider-n_encoders][rem] = 0
            else:
                encoder_decoder_heads[divider-(n_encoders + n_decoders)][rem] = 0
            
            translated = []
            actual = []
            for batch in dataloader:
                pred = model_prediction_ablation(
                    model, 
                    batch, 
                    140, 
                    device, 
                    sp, 
                    encoder_heads=encoder_heads, 
                    decoder_heads=decoder_heads, 
                    encoder_decoder_heads=encoder_decoder_heads
                )
                
                translated.extend(remove_trailing_periods(sp.Decode(pred.detach().cpu().tolist())))
                actual.extend(remove_trailing_periods(sp.Decode(batch['output'].detach().cpu().tolist())))

            bleu_scores[remover] = bleu.corpus_bleu(translated, [actual]).score
        
        max_idx = max(bleu_scores.items(), key=lambda x: x[1])[0]
        last_bleu_score = bleu_scores[max_idx]
        
        removal_order.append(max_idx)
        bleu_scores_removal.append(last_bleu_score)
        print(f"removed idx: {max_idx}, bleu score: {last_bleu_score}")
    
    print(removal_order)
    print(bleu_scores_removal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer Abalation")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--amount", type=int, required=True)
    parser.add_argument("--llm_model_file", type=str, required=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    checkpoint = torch.load(args.llm_model_file, map_location=torch.device(device))
    print(args.llm_model_file)

    splat = args.llm_model_file.split("/")[5].split("_")
    config['N_HEADS'] = int(splat[2])
    config['D_MODEL'] = int(splat[3])
    config['FF_HIDDEN'] = int(splat[4])
    config['N_ENCODERS'] = int(splat[5])
    config['N_DECODERS'] = int(splat[6])

    sp = spm.SentencePieceProcessor(model_file=args.model_file)
    model = build_model(sp, device, checkpoint['model_state_dict'])
    model.eval()

    english_encoded, hindi_encoded = get_encodings(args.dataset, skiprows=550_000, amount=args.amount, sp=sp)
    dataloader = get_dataloaders(sp, english_encoded, hindi_encoded, config['BATCH_SIZE'])
    ablation_studies(model, dataloader, device, sp, heads=config['N_HEADS'], n_decoders=config['N_DECODERS'], n_encoders=config['N_ENCODERS'])
