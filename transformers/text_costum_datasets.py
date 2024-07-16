import os
import torch
import spacy
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import transforms
from collections import defaultdict, Counter
# Download with: python -m spacy download en
root_data = 'D:\Job\Other\pytorch\Flickr8k'
img_data = os.path.join(root_data,'Images')
captions_file = os.path.join(root_data,'captions.txt')
spacy_eng = spacy.load("en_core_web_sm") # load spacy model: an nlp pipeline for English language

class Vocab:
    def __init__(self, freq_threshold):
        # itos: index to string, stoi: string to index ,PAD: padding,
        # SOS: start of sentence, EOS: end of sentence, UNK: unknown
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        # self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.stoi = {v:k for k,v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        """
        Tokenizes English text from a string into a list of strings
        """
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocab(self, sentence_list):
        # Create a dictionary with word frequencies
        word_freq = Counter(self.tokenizer(sentence_list))

        # Create stoi and itos dictionaries
        self.stoi = {word: idx for word, idx in zip(
            word_freq.keys(), range(4, len(word_freq) + 4))}
        self.itos = {idx: word for word, idx in self.stoi.items()}

    # def numericalize(self, text):
    #     tokenized_text = self.tokenizer(text)

    #     return [
    #         self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
    #         for token in tokenized_text
    #     ]
    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)

        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokenized_text
        ]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

class Flickr8kDataset(Dataset):
    def __init__(self, root, captions_file, transform=None, freq_threshold=4):
        self.root = root
        self.df = pd.read_csv(captions_file)  # captions_file
        self.transform = transform
        self.img_names = self.df['image']
        self.captions = self.df['caption']

        # build a vocab
        self.vocab = Vocab(freq_threshold)
        self.vocab.build_vocabulary(self.captions)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.img_names[index]
        img = Image.open(os.path.join(self.root, img_id)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        numericized_caption = [self.vocab.stoi["<SOS>"]] # stoi: sentence to idx, SOS: start of sentence
        numericized_caption += self.vocab.numericalize(caption)
        numericized_caption.append(self.vocab.stoi["<EOS>"]) # EOS: end of sentence

        return img, torch.tensor(numericized_caption)



class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch] # images are 1st element of each item in batch
        imgs = torch.cat(imgs, dim=0) # (batch_size, 3, 256, 256)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)
        return imgs, targets

def get_loader(
    root,
    captions_file,
    transform,
    batch_size=32,
    shuffle=True,
    num_workers=8,
    pin_memory=True
):
    dataset = Flickr8kDataset(root, captions_file, transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=Collate(pad_idx=pad_idx)
    )
    return loader

def main():
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    loader = get_loader(
        root=img_data,
        captions_file=captions_file,
        transform=transform
    )
    for batch_idx, (imgs, captions) in enumerate(loader):
        print(imgs.shape)
        print(captions.shape)
        break

if __name__ == "__main__":
    main()