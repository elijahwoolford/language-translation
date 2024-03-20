import spacy
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader
from typing import List

# Load English and Spanish tokenizers from SpaCy
spacy_eng = spacy.load("en_core_web_sm")
spacy_spa = spacy.load("es_core_news_sm")


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


def tokenize_spa(text):
    return [tok.text for tok in spacy_spa.tokenizer(text)]


def yield_tokens(data_iter: List[str], tokenizer):
    for text in data_iter:
        yield tokenizer(text)


# Define custom Dataset
class TranslationDataset(Dataset):
    def __init__(self, src_file, trg_file, src_tokenizer, trg_tokenizer):
        # Read files
        with open(src_file, encoding='utf-8') as f:
            self.src_lines = f.readlines()
        with open(trg_file, encoding='utf-8') as f:
            self.trg_lines = f.readlines()

        # Assert that both files have the same number of lines
        assert len(self.src_lines) == len(self.trg_lines), "Source and Target must have the same number of lines"

        # Build vocabularies
        self.src_vocab = build_vocab_from_iterator(yield_tokens(self.src_lines, src_tokenizer),
                                                   specials=["<unk>", "<pad>", "<bos>", "<eos>"])
        self.trg_vocab = build_vocab_from_iterator(yield_tokens(self.trg_lines, trg_tokenizer),
                                                   specials=["<unk>", "<pad>", "<bos>", "<eos>"])
        self.src_vocab.set_default_index(self.src_vocab["<unk>"])
        self.trg_vocab.set_default_index(self.trg_vocab["<unk>"])

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_sample = self.src_lines[idx].strip()
        trg_sample = self.trg_lines[idx].strip()

        src_tensor = torch.tensor([self.src_vocab[token] for token in ["<bos>"] + tokenize_eng(src_sample) + ["<eos>"]],
                                  dtype=torch.long)
        trg_tensor = torch.tensor([self.trg_vocab[token] for token in ["<bos>"] + tokenize_spa(trg_sample) + ["<eos>"]],
                                  dtype=torch.long)

        return src_tensor, trg_tensor


# Usage
src_file = 'english_sentences.txt'
trg_file = 'spanish_sentences.txt'
dataset = TranslationDataset(src_file, trg_file, tokenize_eng, tokenize_spa)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through the DataLoader in your training loop
for src, trg in loader:
    # Apply your model on the src and trg tensors
    pass
