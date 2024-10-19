import torch
import torch.nn as nn
#Word embeddings의 차원을 300으로 설정
EMBED_DIMENSION = 300

#restrict word embedding norms. Embedding layer의 weights가 무작정 커지는 것을 막음
EMBED_MAX_NORM = 1

'''
주변 단어들로 center word를 예측하는 모델
주변 단어들의 word embedding vector들의 평균값을 통해 context를 설정하고
그 context를 바탕으로 전체 vocab에 대한 softmax를 계산하여 다음 단어를 예측함
'''
class CBOW_Model(nn.Module):
  def __init__(self, vocab_size: int):
    super(CBOW_Model, self).__init__()
    self.embeddings = nn.Embedding(
      num_embeddings = vocab_size,
      embedding_dim = EMBED_DIMENSION,
      max_norm = EMBED_MAX_NORM     #embedding layer의 weight들이 무작정 커지는 것을 맞아주는 것 --> 성능 높여줌
    )

    '''
    softmax 직전의 linear layer는 context의 word embedding을 입력으로 받고 
    전체 vocabulary 단어들의 unormalized score를 출력한다
    '''
    self.linear = nn.Linear(
      in_features = EMBED_DIMENSION,
      out_features = vocab_size,
    )
  '''
  CBOW 모델이 최종 output을 얻는 방법
  1. one hot vector로 된 token들을 input으로 받고
  2. embedding과 곱해져서 id에 맞는 word embedding vector가 된다
  3. 주변 token들의 word embedding vector를 평균하여 context를 정의
  4. context와 vocab 내 모든 단어들을 dot product하여 각 vocab들의 점수를 계산한다
  '''  
  def forward(self, inputs_):
    x = self.embeddings(inputs_)
    x = x.mean(axis=1)
    x = self.linear(x)
    return x


'''
중심 단어를 통해 주변 단어들을 예측하는 모델
'''
class SkipGram_Model(nn.Module):
    def __init__(self, vocab_size: int):
        super(SkipGram_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        '''
        linear layer는 중심단어의 word embedding을 통해 
        전체 vocab words들이 context word가 될 score를 계산해줌
        '''
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )
    
    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = self.linear(x)
        return x




