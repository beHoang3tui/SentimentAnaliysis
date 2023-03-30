import torch

from utils.utils import get_model, one_hot_word, predict_text

import argparse

parser = argparse.ArgumentParser(description='Choose option of data')
parser.add_argument('-m', '--model', type=str, default='birnn', help='lstm or birnn')
parser.add_argument('-w', '--weight', type=str, default='weight/state_dict.pt', help='weight of model')
args = parser.parse_args()

torch.set_grad_enabled(False)

device = ('cpu', 'cuda')[torch.cuda.is_available()]
vocab = one_hot_word()
num_layers = 2
vocab_size = len(vocab) + 1 # extra 1 for padding
hidden_dim = 256
embedding_dim = 64
output_dim = 3
labels = ('negative', 'neutral', 'positive')

model = get_model(args.model, num_layers, vocab_size, hidden_dim, embedding_dim, output_dim).to(device)
model.load_state_dict(torch.load(args.weight, map_location=torch.device(device)))

def predict(text):
    return labels[predict_text(model, text, vocab, device)]

if __name__ == '__main__':
    text = 'Tốt lắm'
    print(predict(text))