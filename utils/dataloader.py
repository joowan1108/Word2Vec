import torch
from functools import partial
from torch.utils.data import DataLoader
from torchtext.data import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import PennTreebank,EnWik9

#utils에서 정의해둔 constants.py에서 일부 constants 가져오기
from utils.constants import (
    CBOW_N_WORDS,
    SKIPGRAM_N_WORDS,
    MIN_WORD_FREQUENCY,
    MAX_SEQUENCE_LENGTH,
)

#text를 token으로 바꿔줄 tokenizer 가져오기
def get_english_tokenizer():
  tokenizer = get_tokenizer("basic_english", language='en')
  return tokenizer

#학습할 data 가져오기
def get_data_iterator(ds_name, ds_type, data_dir):
  if ds_name == "PennTreebank":
    data_iter = PennTreebank(root=data_dir, split=(ds_type))
  data_iter = to_map_style_dataset(data_iter)
  return data_iter

def build_vocab(data_iter, tokenizer):
  '''
  data에 tokenizer 적용하고 이상한 단어들은 unk token으로 바꿈. 
  이때 모든 단어들에 대해서 하는 것이 아니라 MIN_WORD_FREQUENCY보다 많이 등장한 단어들에 대해서만 적용
  '''
  vocab = build_vocab_from_iterator(
    map(tokenizer, data_iter),
    specials = ["<unk>"],
    min_freq = MIN_WORD_FREQUENCY
  )
  vocab.set_default_index(vocab["<unk>"])
  return vocab

def collate_cbow(batch, text_pipeline):
  '''
  batch(paragraph text)를 mini batch들로 쪼개기 위해 dataloader의 collate_fn argument을 사용
  mini batch를 cbow에 적용하려면 우선 text를 전처리하고 window size에 따라 중간 단어와 context 단어를 따로따로 다른 list에 넣어야함
  이를 collate 함수를 따로 구현하여 적용 --> 즉 algorithm에 맞게 batch를 생성하는 함수
  '''
  batch_input, batch_output = [],[]
  for text in batch:
    #text_pipeline = lambda x: vocab(tokenizer(x)) --> text를 token으로 변환하고 id를 encoding하여 vocab 생성
    text_token_ids = text_pipeline(text)
    #너무 짧은 text는 사용하지 않음 (padding 사용 x)
    if len(text_token_ids) < CBOW_N_WORDS*2+1:
      continue
    #너무 길면, truncate
    if MAX_SEQUENCE_LENGTH:
      text_token_ids = text_token_ids[:MAX_SEQUENCE_LENGTH]

    for idx in range(len(text_token_ids)-2*CBOW_N_WORDS):
      #text의 처음부터 끝까지 window를 옮김
      token_id_sequence = text_token_ids[idx:(idx+CBOW_N_WORDS*2+1)]
      #output은 target word
      output = token_id_sequence.pop(CBOW_N_WORDS)
      #나머지는 context
      input_ = token_id_sequence

      #batch에 저장
      batch_input.append(input_)
      batch_output.append(output)
    
  batch_input = torch.tensor(batch_input, dtype=torch.long)
  batch_output = torch.tensor(batch_output, dtype=torch.long)
  return batch_input, batch_output
  
'''
skipgram을 위한 collate_fn 함수
'''
def collate_skipgram(batch, text_pipeline):
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < SKIPGRAM_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - SKIPGRAM_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + SKIPGRAM_N_WORDS * 2 + 1)]
            input_ = token_id_sequence.pop(SKIPGRAM_N_WORDS)
            outputs = token_id_sequence

            for output in outputs:
                batch_input.append(input_)
                batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output




def get_dataloader_and_vocab(model_name, ds_name, ds_type, data_dir, batch_size, shuffle, vocab=None):
  data_iter = get_data_iterator(ds_name, ds_type, data_dir)
  tokenizer = get_english_tokenizer()

  if not vocab:
      vocab = build_vocab(data_iter, tokenizer)
        
  text_pipeline = lambda x: vocab(tokenizer(x))

  if model_name == "cbow":
    collate_fn = collate_cbow
  elif model_name == "skipgram":
    collate_fn = collate_skipgram
  else:
    raise ValueError("Cbow 또는 Skipgram만 허용")

  dataloader = DataLoader(
    data_iter,
    batch_size=batch_size,
    shuffle=shuffle,
    collate_fn=partial(collate_fn, text_pipeline=text_pipeline),
  )
  return dataloader, vocab  







