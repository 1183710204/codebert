import json
import random

from torch import nn
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_scheduler
from transformers import BertForMaskedLM, RobertaTokenizer, T5ForConditionalGeneration
import torch
import os
import math
from early_stopping import EarlyStopping
from slices2vec import load_pickle

data_dir = 'C:\\Users\\wkr\\Desktop\\dong\\real'
# model_dir = 'C:\\Users\\wkr\\Desktop\\dong\\model\\23-10-08-prompt.pt'
save_dir = 'D:\\dong\\model\\codebert_nodetype.pt'
train_dir = 'D:/dong/data/vul_train.json'
eval_dir = 'D:/dong/data/small_eval.json'


class PROMPTEmbedding(nn.Module):
    def __init__(self, wte: nn.Embedding, n_tokens: int = 10, random_range: float = 0.5,
                 initialize_from_vocab: bool = True):
        super(PROMPTEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(
            self.initialize_embedding(wte, n_tokens, random_range, initialize_from_vocab))

    def initialize_embedding(self, wte: nn.Embedding, n_tokens: int = 10, random_range: float = 0.5,
                             initialize_from_vocab: bool = True):
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(wte.weight.size(1), n_tokens).uniform_(-random_range, random_range)

    def forward(self, tokens):
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        return torch.cat([learned_embedding, input_embedding], 1)


device = torch.device("cuda")
torch.cuda.empty_cache()
model = AutoModelForMaskedLM.from_pretrained('D://dong//codebert-base').to(device)
# model = torch.load(save_dir).to(device)
tokenizer = AutoTokenizer.from_pretrained('D://dong//codebert-base')
yes_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('bad'))[0]
no_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('good'))[0]

# MAX_LENGTH = 500
# WORD_NUM = 20
seq_len = 50
token_num = 25
batch_size = 8
n_tokens = 10
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
token_linear = nn.Linear(token_num, 1).to(device)
# prompt_emb = PROMPTEmbedding(model.get_input_embeddings(),
#                              n_tokens=n_tokens,
#                              initialize_from_vocab=True)
# model.set_input_embeddings(prompt_emb)

# unfreeze_layers = ['11','lm_head']
# for name, param in model.named_parameters():
#     param.requires_grad = False
#     for ele in unfreeze_layers:
#         if ele in name:
#             param.requires_grad = True
#             break
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.size())

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=200,
    num_training_steps=600 * 100 // batch_size
)


def train(input_ids, attention_mask, labels):
    # tp = 0
    # tn = 0
    # fp = 0
    # fn = 0
    input_ids = input_ids.transpose(1, 0)
    attention_mask = attention_mask.transpose(1, 0)
    seq_emb = torch.zeros((seq_len, batch_size, 768)).to(device)
    for i in range(seq_len):
        emb = model.base_model.embeddings(input_ids[i])
        seq_emb[i] = token_linear(emb.transpose(2,1)).squeeze(-1)
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
    attention_mask = attention_mask.transpose(1, 0)
    attention_mask, _ = torch.max(attention_mask, dim=-1)
    outputs = model(inputs_embeds=seq_emb.transpose(1, 0), attention_mask=attention_mask,
                    labels=labels.long())
    # logits = outputs.logits
    # yes_token_logits = logits[:, :, yes_token_id]
    # no_token_logits = logits[:, :, no_token_id]
    # results = torch.stack((no_token_logits, yes_token_logits), 2)
    # y_softmax = nn.Softmax(dim=2)
    # y_results = y_softmax(results)
    # action_dist = torch.distributions.Categorical(y_results)
    # action = action_dist.sample()
    # action = action.unsqueeze(dim=2)
    # probs_one = y_results.gather(2, action).squeeze()
    # action = action.squeeze()
    # p_mul = 0
    # row_idxs, col_idxs = torch.where(labels != -100)
    # row_idxs = row_idxs.tolist()
    # col_idxs = col_idxs.tolist()
    # for row_idx, col_idx in zip(row_idxs, col_idxs):
    #     y_hat = bool(action[row_idx, col_idx])
    #     p_mul = p_mul + torch.log(probs_one[row_idx, col_idx])
    #     # p_mul=p_mul+torch.log(y_results[row_idx, col_idx,1])
    #     y = bool(labels[row_idx, col_idx] == yes_token_id)
    #     if y_hat and y:
    #         tp += 1
    #     elif not y_hat and not y:
    #         tn += 1
    #     elif y_hat and not y:
    #         fp += 1
    #     else:
    #         fn += 1
    # iou = tp / (tp + fn + fp) if tp + fn + fp != 0 else 0
    # loss = -p_mul *iou
    # loss = torch.tensor(1.0-iou)
    # loss.requires_grad_(True)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    return loss.item()


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
            input_id = data['orig_code']
            line_type = data['line_type']
            for index, node_type in enumerate(line_type):
                input_id[index] = node_type + ' ' + input_id[index]
            fine_label = [no_token_id]*seq_len
            for label in data['vul_lines']:
                if -1 < label < seq_len:
                    fine_label[label] = yes_token_id
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
    data['fine_labels'] = torch.tensor(fine_labels)
    data['coarse_labels'] = torch.stack(coarse_labels)
    data['labels_mask'] = torch.stack(labels_mask)
    data['data_size'] = len(input_ids)
    return data


# def load_code(data, num):
#     block_size = len(data)
#     all_input_ids = []
#     all_attention_mask = []
#     all_labels = []
#     for index in tqdm(range(min(block_size, num))):
#         x = data.iloc[index]
#         label = x['label']
#         # if label[0]==0:
#         #     continue
#         code_with_label = [no_token_id] * len(x['map_code'])
#         for _label in label:
#             if _label != 0:
#                 code_with_label[_label - 1] = yes_token_id
#         input_ids = [tokenizer.mask_token_id] * n_tokens
#         labels = [-100] * n_tokens
#         for code, label_token_id in zip(x['map_code'], code_with_label):
#             # mask0
#             # token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(code))
#             token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(code)) + [tokenizer.mask_token_id]
#             # # mask1
#             # token_ids = tokenizer.convert_tokens_to_ids(
#             #     tokenizer.tokenize('Code is')) + [tokenizer.mask_token_id] + tokenizer.convert_tokens_to_ids(
#             #     tokenizer.tokenize('vulnerable.')) + [tokenizer.sep_token_id] + tokenizer.convert_tokens_to_ids(
#             #     tokenizer.tokenize(code)) + [tokenizer.sep_token_id]
#             # # mask2
#             # token_ids = tokenizer.convert_tokens_to_ids(
#             #     tokenizer.tokenize('It is a')) + [tokenizer.mask_token_id] + tokenizer.convert_tokens_to_ids(
#             #     tokenizer.tokenize('code.')) + tokenizer.convert_tokens_to_ids(
#             #     tokenizer.tokenize(code))
#             # # mask3
#             # token_ids = tokenizer.convert_tokens_to_ids(
#             #     tokenizer.tokenize('Is code safe?Answer is')) + [tokenizer.mask_token_id] + [
#             #                 tokenizer.sep_token_id] + tokenizer.convert_tokens_to_ids(
#             #     tokenizer.tokenize(code)) + [tokenizer.sep_token_id]
#
#             label = [-100] * (len(token_ids) - 1) + [label_token_id]
#             # label = [label_token_id] * len(token_ids)
#             if len(input_ids) + len(token_ids) <= MAX_LENGTH:
#                 input_ids += token_ids
#                 labels += label
#         attention_mask = [1] * len(input_ids)
#         length = len(input_ids)
#         if length < MAX_LENGTH:
#             input_ids += [tokenizer.pad_token_id] * (MAX_LENGTH - length)
#             attention_mask += [0] * (MAX_LENGTH - length)
#             labels += [-100] * (MAX_LENGTH - length)
#         all_input_ids.append(input_ids)
#         all_attention_mask.append(attention_mask)
#         all_labels.append(labels)
#     all_input_ids = torch.LongTensor(all_input_ids)
#     all_attention_mask = torch.LongTensor(all_attention_mask)
#     all_labels = torch.LongTensor(all_labels)
#     return all_input_ids, all_attention_mask, all_labels


def evaluate(input_ids, attention_mask, labels):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    tp_b = 0
    tn_b = 0
    fp_b = 0
    fn_b = 0
    model.eval()
    with torch.no_grad():
        num = int(input_ids.size(0))
        for i in range(num // batch_size):
            batch_input_ids = input_ids[i * batch_size:(i + 1) * batch_size]
            batch_attention_mask = attention_mask[i * batch_size:(i + 1) * batch_size]
            batch_labels = labels[i * batch_size:(i + 1) * batch_size]
            y_num = [0] * batch_size
            label_num = [0] * batch_size
            batch_input_ids = batch_input_ids.transpose(1, 0)
            batch_attention_mask = batch_attention_mask.transpose(1, 0)
            seq_emb = torch.zeros((seq_len, batch_size, 768)).to(device)
            for j in range(seq_len):
                emb = model.base_model.embeddings(batch_input_ids[j])
                seq_emb[j] = token_linear(emb.transpose(2, 1)).squeeze(-1)
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
            batch_attention_mask = batch_attention_mask.transpose(1, 0)
            batch_attention_mask, _ = torch.max(batch_attention_mask, dim=-1)
            outputs = model(inputs_embeds=seq_emb.transpose(1, 0), attention_mask=batch_attention_mask)
            logits = outputs.logits
            yes_token_logits = logits[:, :, yes_token_id]
            no_token_logits = logits[:, :, no_token_id]
            results = torch.stack((no_token_logits, yes_token_logits), 2)
            y_softmax = nn.Softmax(dim=2)
            y_results = y_softmax(results)
            action_dist = torch.distributions.Categorical(y_results)
            action = action_dist.sample()
            row_idxs, col_idxs = torch.where(batch_labels != -100)
            row_idxs = row_idxs.tolist()
            col_idxs = col_idxs.tolist()
            for row_idx, col_idx in zip(row_idxs, col_idxs):
                y_hat = bool(action[row_idx, col_idx])
                y = bool(batch_labels[row_idx, col_idx] == yes_token_id)
                if y_hat and y:
                    tp += 1
                    y_num[row_idx] = 1
                    label_num[row_idx] = 1
                elif not y_hat and not y:
                    tn += 1
                elif y_hat and not y:
                    fp += 1
                    y_num[row_idx] = 1
                else:
                    fn += 1
                    label_num[row_idx] = 1
            for j in range(batch_size):
                if y_num[j] == 1 and label_num[j] == 1:
                    tp_b += 1
                elif y_num[j] == 0 and label_num[j] == 0:
                    tn_b += 1
                elif y_num[j] == 1 and label_num[j] == 0:
                    fp_b += 1
                else:
                    fn_b += 1
        a = (tp_b + tn_b) / (tp_b + tn_b + fp_b + fn_b)
        p = tp_b / (tp_b + fp_b) if tp_b + fp_b != 0 else 0
        r = tp_b / (tp_b + fn_b) if tp_b + fn_b != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        iou = tp / (tp + fn + fp) if tp + fn + fp != 0 else 0
        print('粗粒度 ', tp_b, tn_b, fp_b, fn_b)
        print('细粒度 ', tp, tn, fp, fn)
        print(round(a, 4), round(p, 4), round(r, 4), round(f1, 4), round(iou, 4))


def main():
    # train_data = load_pickle(os.path.join(data_dir, 'train', 'real_source_total_blocks.pkl'))
    # test_data = load_pickle(os.path.join(data_dir, 'target', 'real_source_total_blocks.pkl'))
    # train_data = load_pickle(os.path.join(data_dir, 'train', 'source_total_blocks_c.pkl'))
    # test_data = load_pickle(os.path.join(data_dir, 'target', 'source_total_blocks_c.pkl'))
    # train_input_ids, train_attention_mask, train_labels = load_code(train_data, 600)
    # test_input_ids, test_attention_mask, test_labels = load_code(test_data, 1600)
    # train_data = load_json(train_dir)
    eval_data = load_json(eval_dir)
    # train_input_ids = train_data['input_ids']
    # train_attention_mask = train_data['attention_mask']
    # train_labels = train_data['fine_labels']
    test_input_ids = eval_data['input_ids']
    test_attention_mask = eval_data['attention_mask']
    test_labels = eval_data['fine_labels']
    early_stopping = EarlyStopping(patience=1000, verbose=True)
    # evaluate(test_input_ids.to(device), test_attention_mask.to(device), test_labels.to(device))
    for n in tqdm(range(100)):
        loss = 0
        model.train()
        for i in range(int(len(train_input_ids) / batch_size)):
            batch_input_ids = train_input_ids[i * batch_size:(i + 1) * batch_size]
            batch_attention_mask = train_attention_mask[i * batch_size:(i + 1) * batch_size]
            batch_labels = train_labels[i * batch_size:(i + 1) * batch_size]

            loss += train(
                batch_input_ids.to(device),
                batch_attention_mask.to(device),
                batch_labels.to(device)
            )
        evaluate(test_input_ids.to(device), test_attention_mask.to(device), test_labels.to(device))
        early_stopping(loss, model, save_dir)


if __name__ == '__main__':
    main()
