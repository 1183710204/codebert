from __future__ import print_function
from gensim.models import Word2Vec, word2vec
import pickle
import os
import gc
import os


class DirofCorpus(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self.iter_num = 0

    def __iter__(self):
        print("\nepoch: ", self.iter_num)
        for d in self.dirname:
            for fn in os.listdir(d):
                print("\r" + fn, end='')
                for filename in os.listdir(os.path.join(d, fn)):
                    sample = pickle.load(open(os.path.join(d, fn, filename), 'rb'))[0]
                    yield sample

        self.iter_num += 1


def generate_w2vModel(decTokenFlawPath, w2vModelPath, size=30, alpha=0.008, window=5, min_alpha=0.0005, sg=1, hs=0,
                      negative=7, iter=3):
    print("training...")
    model = Word2Vec(sentences=word2vec.TextBCorpus(decTokenFlawPath), size=size, alpha=alpha, window=window, min_count=1,
                     max_vocab_size=None, sample=0.001, seed=1, workers=1, min_alpha=min_alpha, sg=sg, hs=hs,
                     negative=negative, iter=iter)
    model.save(w2vModelPath)


def evaluate_w2vModel(w2vModelPath):
    print("\nevaluating...")
    model = Word2Vec.load(w2vModelPath)
    for sign in ['(', 'icmp', 'func_0', 'i32', '%2']:
        print(sign, ":")
        print(model.most_similar_cosmul(positive=[sign], topn=10))

def load_json(path):
    input_ids = []
    attention_mask = []
    fine_labels = []  # 细粒度
    coarse_labels = []  # 粗粒度
    labels_mask = []
    with open(path, 'r') as file:
        lines = file.readlines()[0]
        dataset = json.loads(lines)
        for data in dataset:
            if data['label'] == 0:
                continue
            input_id = data['orig_code']
            fine_label = torch.zeros(seq_len)
            for label in data['vul_lines']:
                if -1 < label < seq_len:
                    fine_label[label] = 1
            coarse_label = torch.Tensor([data['label']])  # int
            input_encoding = tokenizer(input_id, padding=True, return_tensors="pt")
            input_embedding = input_encoding['input_ids']
            input_mask = input_encoding['attention_mask']
            label_mask = torch.ones(seq_len)
            padding_num = len(input_embedding[0])
            if padding_num > token_num:
                input_embedding = input_embedding[:, :token_num]
                input_mask = input_mask[:, :token_num]
            elif padding_num < token_num:
                pad = nn.ConstantPad2d(padding=(0, token_num - padding_num, 0, 0), value=tokenizer.pad_token_id)
                pad_mask = nn.ConstantPad2d(padding=(0, token_num - padding_num, 0, 0), value=0)
                input_embedding = pad(input_embedding)
                input_mask = pad_mask(input_mask)
            if len(input_embedding) < seq_len:
                label_mask = torch.cat((torch.ones(len(input_embedding)), torch.zeros(seq_len - len(input_embedding))),
                                       dim=0)
                pad = nn.ConstantPad2d(padding=(0, 0, 0, seq_len - len(input_embedding)), value=tokenizer.pad_token_id)
                pad_mask = nn.ConstantPad2d(padding=(0, 0, 0, seq_len - len(input_embedding)), value=0)
                input_embedding = pad(input_embedding)
                input_mask = pad_mask(input_mask)
            elif len(input_embedding) > seq_len:
                input_embedding = input_embedding[:seq_len, :]
                input_mask = input_mask[:seq_len, :]
            input_ids.append(input_embedding)
            attention_mask.append(input_mask)
            fine_labels.append(fine_label)
            coarse_labels.append(coarse_label)
            labels_mask.append(label_mask)
    data = dict()
    data['input_ids'] = torch.stack(input_ids)
    data['attention_mask'] = torch.stack(attention_mask)
    data['fine_labels'] = torch.stack(fine_labels)
    data['coarse_labels'] = torch.stack(coarse_labels)
    data['labels_mask'] = torch.stack(labels_mask)
    data['data_size'] = len(input_ids)
    return data


def main():
    dec_tokenFlaw_path = ['./data/corpus/']

    for iter in [3, 5, 10, 15]:
        w2v_model_path = "D:\dong\module\word2vec.model"
        generate_w2vModel(dec_tokenFlaw_path, w2v_model_path, iter=iter)
        evaluate_w2vModel(w2v_model_path)


if __name__ == "__main__":
    main()