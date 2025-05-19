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

from BPE.bpe import BPEEncoder, BPEDecoder
from TransformerComponents.Encoder import Encoder
from TransformerComponents.Decoder import Decoder
from papers.TransformerComponents.BasePositionalEncoding import PositionalEmbedding
from TransformerComponents.Transformer import Transformer
from TransformerComponents.UtilsLayers import Projection
from TransformerComponents.Optimiser import WarmupAdamOpt

YAML_PATH = "dl-from-scratch/papers/attention_is_all_you_need/config.yaml"
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
        src_encodings = [[sos_token] + sublist + [eos_token] for sublist in src_encodings]
        tgt_encodings = [[sos_token] + sublist for sublist in tgt_encodings]
        output_encodings = [sublist + [eos_token] for sublist in tgt_encodings] 
        full_encoding = list(zip(src_encodings, tgt_encodings, output_encodings))
        full_encoding.sort(key=lambda x: len(x[0])) # sort sequence lengths
        return full_encoding


    def __getitem__(self, idx):
        src_seq, tgt_seq, output_seq = self.paired_encodings[idx]
        src_tensor = torch.tensor(src_seq, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_seq, dtype=torch.long)
        output_tensor = torch.tensor(output_seq, dtype=torch.long)

        src_tensor = F.pad(src_tensor, (0, self.seq_len - src_tensor.size(0)), value=self.pad_token)
        tgt_tensor = F.pad(tgt_tensor, (0, self.seq_len - tgt_tensor.size(0)), value=self.pad_token)
        output_tensor = F.pad(output_tensor, (0, self.seq_len - output_tensor.size(0)), value=self.pad_token)
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
    

def get_encodings(datapath, vocab_path, vocab_file):
    dataset = pd.read_csv(datapath)
    vocab_file_path = Path(vocab_path) / vocab_file
    english_encoded_path = Path(vocab_path) / "english_encoded"
    german_encoded_path = Path(vocab_path) / "german_encoded"
    with open(vocab_file_path, "rb") as f:
        vocab = pickle.load(f)

    bpe_encoder = BPEEncoder(vocab=vocab)
    # Check if encoded files exist
    if english_encoded_path.exists() and german_encoded_path.exists():
        with open(english_encoded_path, "rb") as f:
            english_encoded = pickle.load(f)
        with open(german_encoded_path, "rb") as f:
            german_encoded = pickle.load(f)
    else:
        english_encoded = bpe_encoder.encode(dataset.iloc[:, 1])
        german_encoded = bpe_encoder.encode(dataset.iloc[:, 0])

        # Save encoded data
        with open(english_encoded_path, "wb") as f:
            pickle.dump(english_encoded, f)
        with open(german_encoded_path, "wb") as f:
            pickle.dump(german_encoded, f)

    return english_encoded, german_encoded, vocab

def get_decoder(vocab):
    max_vocab_size = max(vocab.values())
    special_tokens = {max_vocab_size + 1 : "<SOS>",  max_vocab_size + 2 : "<EOS>", max_vocab_size + 3 : "<pad>"}
    bpe_decoder = BPEDecoder(vocab=vocab, special_tokens=special_tokens)
    return bpe_decoder

def get_dataloaders(vocab, english_encoded, german_encoded):
    max_vocab_size = max(vocab.values())
    full_data = LanguageTranslationDataset(seq_length=config['SEQ_LEN'], src_encodings=english_encoded, tgt_encodings=german_encoded, sos_token=max_vocab_size + 1, eos_token=max_vocab_size + 2,
                                        pad_token=max_vocab_size + 3)
    train_data, test_data = random_split(full_data, [config['TRAIN_RATIO'], 1-config['TRAIN_RATIO']])
    train_dataloader = DataLoader(train_data, batch_size=config['BATCH_SIZE'], shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=config['BATCH_SIZE'], shuffle=True, pin_memory=True)
    return train_dataloader, test_dataloader

def build_model(vocab, device):
    vocab_size = len(vocab) + 256 + 3 # 3 special tokens
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


def train(model, max_vocab_size, vocab_size, train_dataloader, test_dataloader, device, bpe_decoder, warmup_steps):
    exp_name = "drop_warm_smoothen"
    num_examples = 10
    if warmup_steps != 0:
        optimiser = WarmupAdamOpt(config['D_MODEL'], warmup_steps, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    else:
        optimiser = torch.optim.Adam(model.parameters(), lr=config['LR'], eps=1e-9)
    
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=max_vocab_size + 3, label_smoothing=0.1).to(device)
    global_step_train = 0
    global_step_test = 0
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
            loss = loss_fn(logits.view(-1, vocab_size), target_indices.view(-1))
            batch_loss += loss.item()
            
            if warmup_steps != 0:
                optimiser.optimiser.zero_grad()
            else:
                optimiser.zero_grad()
            
            loss.backward()
            optimiser.step()
            global_step_train += 1

        losses.append(batch_loss / len(batch_train))


        model.eval()
        val_loss = 0
        batch_test = iter(test_dataloader)
        sample_taken = False
        for data in batch_test:
            with torch.no_grad():
                target_indices = data['output'].to(device)
                if not sample_taken and epoch % 100 == 0:
                    pred = model_prediction(model, data, config['SEQ_LEN'], device, max_vocab_size + 1, max_vocab_size + 2, max_vocab_size + 3)
                    ints = torch.randint(low=0, high=pred.size(0), size=(num_examples,))
                    pred = pred[ints, :]
                    decoded = [decoded.replace("<pad>", "") for decoded in bpe_decoder.decode(pred.detach().cpu().tolist())]
                    actual_decoded = [decoded.replace("<pad>", "") for decoded in bpe_decoder.decode(target_indices[ints, :].detach().cpu().tolist())]
                    
                    comparison_text = []
                    for j in range(len(decoded)):
                        comparison_text.append([decoded[j], actual_decoded[j]])

                    sentences[epoch] = comparison_text
                    global_step_test += 1
                    sample_taken = True

                encoder_input = data['src'].to(device) # B x seq_len
                tgt_input = data['tgt'].to(device) # B x seq_len
                encoder_mask = data['encoder_mask'].to(device) # B x 1 x 1 x seq_len
                decoder_mask = data['decoder_mask'].to(device) # B x 1 x seq_len x seq_len
                logits = model(encoder_input,  tgt_input, encoder_mask=encoder_mask, decoder_mask=decoder_mask)
                loss = loss_fn(logits.view(-1, vocab_size), target_indices.view(-1))
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
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--vocab_file", type=str, required=True)
    parser.add_argument("--warmup", type=int, required=False, default=500)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    english_encoded, german_encoded, vocab = get_encodings(args.dataset, args.vocab_path, args.vocab_file)
    bpe_decoder = get_decoder(vocab)
    train_dataloader, test_dataloader = get_dataloaders(vocab, english_encoded, german_encoded)
    model = build_model(vocab, device)
    if torch.cuda.device_count() > 1:
        pass
    
    vocab_size = len(vocab) + 256 + 3 # Length of merges dictionary, + 256 base utf + 3 special tokens
    train(model, max(vocab.values()), vocab_size, train_dataloader, test_dataloader, device, bpe_decoder, args.warmup)
