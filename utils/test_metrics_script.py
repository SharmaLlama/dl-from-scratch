from collections import OrderedDict
import sentencepiece as spm
import torch
import json 
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import yaml
import random
import itertools
import pandas as pd
import torch.nn.functional as F
import argparse
from collections import Counter
from metrics import corpus_bleu, brevity_penality

from TransformerComponents.Encoder import Encoder
from TransformerComponents.Decoder import Decoder
from TransformerComponents.PE import PositionalEmbedding
from TransformerComponents.Transformer import Transformer
from TransformerComponents.UtilsLayers import Projection
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

YAML_PATH = "dl-from-scratch/papers/Attention Is All You Need/config.yaml"
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

            # Ensure the sequence length does not exceed `seq_len`
            if src_tensor.size(0) > self.seq_len:
                src_tensor = src_tensor[:self.seq_len]  # Crop excess tokens
            if tgt_tensor.size(0) > self.seq_len:
                tgt_tensor = tgt_tensor[:self.seq_len]  # Crop excess tokens
            if output_tensor.size(0) > self.seq_len:
                output_tensor = output_tensor[:self.seq_len]  # Crop excess tokens

            # **Pad sequences to `seq_len` if they are shorter**
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
        model.load_state_dict(state_dict)
    elif torch.cuda.device_count() > 1:
        model.module.initialise()
    else: 
        model.initialise() 
    return model

def get_data(datapath, skiprows, amount, sp):
    english_sentences = pd.read_table(Path(datapath) /  "english_small.txt",  header=None, skiprows=skiprows, nrows=amount)
    english_encoded = sp.encode_as_ids(english_sentences.iloc[:, 0].to_list())
    with open(datapath / "test.json", 'r') as f:
        ref_sentences = json.load(f)
    ref_sentences = ref_sentences[0:amount]
    encoding_hindi = list(itertools.chain.from_iterable(ref_sentences))
    hindi_encoded = sp.encode_as_ids(encoding_hindi)[:amount]
    return english_encoded, hindi_encoded, ref_sentences


def get_dataloaders(sp, english_encoded, tgt_encoded, amount):
    full_data = LanguageTranslationDataset(seq_length=config['SEQ_LEN'], src_encodings=english_encoded, tgt_encodings=tgt_encoded, sos_token=sp.bos_id(), eos_token=sp.eos_id(),
                                        pad_token=sp.pad_id())
    dataloader = DataLoader(full_data, batch_size=amount, pin_memory=True, shuffle=False)
    return dataloader


def model_prediction(model, batch, max_len, device, sos_token, eos_token, pad_token):
    underlying_model = model.module if hasattr(model, 'module') else model

    encoder_input = batch['src'].to(device)  # B x seq_len
    encoder_mask = batch['encoder_mask'].to(device)  # B x 1 x 1 x seq_len
    encoder_output = underlying_model.encode(encoder_input, encoder_mask)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer Training")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--amount", type=int, required=False, default=500)
    parser.add_argument("--llm_model_file", type=str, required=False, default="")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.llm_model_file != "":
        checkpoint = torch.load(args.llm_model_file, map_location=torch.device(device))
    else:
        checkpoint = {'model_state_dict' : None, 'optimiser_state_dict' : None}

    sp = spm.SentencePieceProcessor(model_file=args.model_file)
    model = build_model(sp, device, checkpoint['model_state_dict'])
    num_c = Counter({1: 0, 2: 0, 3: 0, 4:0})
    denom_c = Counter({1: 0, 2: 0, 3: 0, 4:0})
    ref_len = 0
    cand_len = 0
    
    english_encoded, hindi_encoded, ref_sentences = get_data(args.dataset, skiprows=550_000, amount=args.amount, sp=sp)
    dataloader = get_dataloaders(sp, english_encoded, hindi_encoded, config['BATCH_SIZE'])
    
    for idx, batch in enumerate(dataloader):
        pred = model_prediction(model, batch, config['SEQ_LEN'], device, sp.bos_id(), sp.eos_id(), sp.pad_id())
        decoded = sp.decode(pred.detach().cpu().tolist())
        num, denom, cand, ref = corpus_bleu(ref_sentences[(idx) * args.amount: (idx + 1) * args.amount], decoded, raw_values=True) 
        num_c += num
        denom_c += denom
        cand_len += cand
        ref_len += ref

    bp = brevity_penality(cand_len, ref_len)
    pn = [num_c[i] / denom_c[i] if denom_c[i] > 0 else 0 for i in range(1, 5)]
    log_sum = sum([w * np.log(p) if p > 0 else 0 for w,p in zip((0.25, 0.25, 0.25, 0.25), pn)])
    bs = bp * np.exp(log_sum)
    print(f"The BLEU score for the test set is: {bs}")