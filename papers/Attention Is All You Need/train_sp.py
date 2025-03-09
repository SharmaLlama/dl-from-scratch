import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split
import torch.nn.functional as F
import pickle
import pandas as pd
from pathlib import Path
import argparse
import yaml
import os
import sentencepiece as spm

from TransformerComponents.Encoder import Encoder
from TransformerComponents.Decoder import Decoder
from TransformerComponents.PE import PositionalEmbedding
from TransformerComponents.Transformer import Transformer
from TransformerComponents.UtilsLayers import Projection
from TransformerComponents.Optimiser import WarmupAdamOpt

YAML_PATH = "../papers/Attention Is All You Need/config.yaml"
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
        

def get_encodings(datapath, model_file):
    english_sentences = pd.read_table(Path(datapath) /  "english_small.txt",  header=None)
    hindi_sentences = pd.read_table(Path(datapath) /  "hindi_small.txt",  header=None)
    sp = spm.SentencePieceProcessor(model_file=model_file)
    english_encoded = sp.encode_as_ids(english_sentences.iloc[:, 0].to_list())
    hindi_encoded = sp.encode_as_ids(hindi_sentences.iloc[:, 0].to_list())
    return english_encoded, hindi_encoded, sp


def get_dataloaders(sp, english_encoded, tgt_encoded):
    full_data = LanguageTranslationDataset(seq_length=config['SEQ_LEN'], src_encodings=english_encoded, tgt_encodings=tgt_encoded, sos_token=sp.bos_id(), eos_token=sp.eos_id(),
                                        pad_token=sp.pad_id())
    train_data, test_data = random_split(full_data, [config['TRAIN_RATIO'], 1-config['TRAIN_RATIO']])
    train_dataloader = DataLoader(train_data, batch_size=config['BATCH_SIZE'], shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=config['BATCH_SIZE'], shuffle=True, pin_memory=True)
    return train_dataloader, test_dataloader

def build_model(sp, device):
    vocab_size = sp.vocab_size()
    encoder_transformer = Encoder(config['N_ENCODERS'], config['N_HEADS'], config['D_MODEL'], config['D_MODEL'] // config['N_HEADS'], config['D_MODEL'] // config['N_HEADS'], config['FF_HIDDEN'], config['DROPOUT'])
    decoder_transformer = Decoder(config['N_DECODERS'], config['N_HEADS'], config['D_MODEL'], config['D_MODEL'] // config['N_HEADS'], config['D_MODEL'] // config['N_HEADS'], config['FF_HIDDEN'], config['DROPOUT'])
    src_embeddings = PositionalEmbedding(vocab_size, config['D_MODEL'], config['SEQ_LEN'], config['DROPOUT'])
    tgt_embeddings = PositionalEmbedding(vocab_size, config['D_MODEL'], config['SEQ_LEN'], config['DROPOUT'])
    projection = Projection(config['D_MODEL'], vocab_size)
    model = Transformer(encoder_transformer, decoder_transformer, src_embeddings, tgt_embeddings, projection).to(device)
    model.initialise()

    return model


def model_prediction(model, batch, max_len, device, sos_token, eos_token, pad_token):
    encoder_input = batch['src'].to(device) # B x seq_len
    encoder_mask = batch['encoder_mask'].to(device) # B  x 1 x 1 x seq_len
    encoder_output = model.encode(encoder_input, encoder_mask)
    B = encoder_input.size(0)
    decoder_input = torch.full((B, max_len), pad_token).to(device)
    decoder_input[: , 0] = sos_token
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for t in range(max_len - 1):
        subsequent_mask = torch.tril(torch.ones((max_len, max_len), dtype=torch.int)).expand(B, -1, -1).to(device) # shape: (B, max_len, max_len)
        other_mask =(decoder_input != pad_token).int().unsqueeze(1) # (B, 1, max_len)
        decoder_mask = (subsequent_mask & other_mask).unsqueeze(1).to(device)
        out = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
        prediction = model.proj(out) # Expected shape: (B, max_len, vocab_size)
        next_tokens = torch.argmax(prediction[:, t, :], dim=-1) # shape: (B, )
        next_tokens = torch.where(finished, pad_token, next_tokens)

        decoder_input[:, t + 1] = next_tokens
        finished |= (next_tokens == eos_token)

        if finished.all():
          break

    return decoder_input


def train(model, sp, train_dataloader, test_dataloader, device, warmup_steps):
    exp_name = "hindi_drop_warm_smoothen"
    num_examples = 10
    if warmup_steps != 0:
        optimiser = WarmupAdamOpt(config['D_MODEL'], warmup_steps, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    else:
        optimiser = torch.optim.Adam(model.parameters(), lr=config['LR'], eps=1e-9)
    
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=sp.pad_id(), label_smoothing=0.1).to(device)
    losses = []
    test_losses = []
    sentences = {}
    base_model_dir = f"/srv/scratch/z3547870/Models/{exp_name}"
    model_dir = base_model_dir
    if os.path.exists(base_model_dir):
        counter = 1 
        while os.path.exists(model_dir):
            model_dir = f"{base_model_dir}_{counter}"
            counter += 1

    os.makedirs(model_dir, exist_ok=True)  # Ensure the directory exists
    
    for epoch in range(1, config['NUM_EPOCHS'] + 1):
        model.train()
        batch_train = iter(train_dataloader)
        batch_loss = 0
        for data in batch_train:
            target_indices = data['output'].to(device) # B x seq_len

            encoder_input = data['src'].to(device) # B x seq_len
            tgt_input = data['tgt'].to(device) # B x seq_len
            encoder_mask = data['encoder_mask'].to(device) # B x 1 x 1 x seq_len
            decoder_mask = data['decoder_mask'].to(device) # B x 1 x seq_len x seq_len
            logits = model(encoder_input,  tgt_input, encoder_mask=encoder_mask, decoder_mask=decoder_mask)
            loss = loss_fn(logits.view(-1, sp.vocab_size()), target_indices.view(-1))
            batch_loss += loss.item()
            
            if warmup_steps != 0:
                optimiser.optimiser.zero_grad()
            else:
                optimiser.zero_grad()
            
            loss.backward()
            optimiser.step()

        losses.append(batch_loss / len(batch_train))


        model.eval()
        val_loss = 0
        batch_test = iter(test_dataloader)
        sample_taken = False
        for data in batch_test:
            with torch.no_grad():
                target_indices = data['output'].to(device)
                encoder_input = data['src'].to(device)
                if not sample_taken and epoch % 100 == 0:
                    pred = model_prediction(model, data, config['SEQ_LEN'], device, sp.bos_id(), sp.eos_id(), sp.pad_id())
                    ints = torch.randint(low=0, high=pred.size(0), size=(num_examples,))
                    pred = pred[ints, :]
                    decoded = sp.decode(pred.detach().cpu().tolist())
                    actual_decoded = sp.decode(target_indices[ints, :].detach().cpu().tolist())
                    source_sentence = sp.decode(encoder_input[ints, :].detach().cpu().to_list())
                    comparison_text = list(zip(source_sentence, actual_decoded, decoded))
                    sentences[epoch] = comparison_text
                    sample_taken = True

                encoder_input = data['src'].to(device) # B x seq_len
                tgt_input = data['tgt'].to(device) # B x seq_len
                encoder_mask = data['encoder_mask'].to(device) # B x 1 x 1 x seq_len
                decoder_mask = data['decoder_mask'].to(device) # B x 1 x seq_len x seq_len
                logits = model(encoder_input,  tgt_input, encoder_mask=encoder_mask, decoder_mask=decoder_mask)
                loss = loss_fn(logits.view(-1, sp.vocab_size()), target_indices.view(-1))
                val_loss += loss.item()

        test_losses.append(val_loss / len(batch_test))

        if epoch % 100 == 0:
            model_filename = f"{model_dir}/Model_{epoch}"
            torch.save({
                'epoch': epoch,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                }, model_filename)
    
    with open(f"/srv/scratch/z3547870/experiments/{exp_name}_train_loss.pkl", "wb") as f:
        pickle.dump(losses, f)    
        
    with open(f"/srv/scratch/z3547870/experiments/{exp_name}_test_loss.pkl", "wb") as f:
        pickle.dump(test_losses, f)    
    
    with open(f"/srv/scratch/z3547870/experiments/{exp_name}_sentences.pkl", "wb") as f:
        pickle.dump(sentences, f)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer Training")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--warmup", type=int, required=False, default=500)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    english_encoded, hindi_encoded, sp = get_encodings(args.dataset, args.model_file)
    train_dataloader, test_dataloader = get_dataloaders(sp, english_encoded, hindi_encoded)
    model = build_model(sp, device)
    train(model,sp, train_dataloader, test_dataloader, device, args.warmup)
