# 开发时间 2023/11/17 14:08
# 获得完整函数的CodeBERT表示
import os
import shutil
import time

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# 处理整个函数的codeBert表示
def process_func():
    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained("D://dong//codebert-base")
    model = AutoModel.from_pretrained("D://dong//codebert-base")
    print("Processing...")

    path = "D:\\OPENSSL_withPDG"
    for pj in tqdm(os.listdir(path)):
        pj_path = path + "\\" + pj + "\\func_related"
        for func_file in os.listdir(pj_path):
            with open(pj_path+"\\"+func_file,'r') as f1:
                lines = f1.readlines()
                func_all_string = ""
                for line in lines:
                    func_all_string = func_all_string+line
            f1.close()
            SC_Tokens = tokenizer.tokenize(func_all_string)
            Tokens = [tokenizer.cls_token]+SC_Tokens+[tokenizer.eos_token]
            if len(Tokens)>=512:
                Tokens = Tokens[0:511]+[tokenizer.eos_token]
            tokens_id = tokenizer.convert_tokens_to_ids(Tokens)
            tokens_tensor = torch.tensor(tokens_id)[None, :]
            with torch.no_grad():
                try:
                    results = model(tokens_tensor)
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        print('WARNING: out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
            last_state = results.last_hidden_state
            func_repr = last_state[0][0]
            torch.save(func_repr,path+"\\"+pj+"\\func_codeBert\\"+func_file.split(".")[0]+".pt")

# 处理PDG每一个节点的codeBert表示
def process_line():
    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained("D://dong//codebert-base")
    model = AutoModel.from_pretrained("D://dong//codebert-base")
    print("Processing...")
    path = "D:\\OPENSSL_withPDG"
    for pj in tqdm(os.listdir(path)):
        pj_path = path+"\\"+pj
        for file in os.listdir(pj_path):
            if file.endswith(".c"):
                c_file = file
        c_file_path = pj_path+"\\"+c_file
        with open(c_file_path,'r') as f1:
            c_lines = f1.readlines()
        f1.close()
        for index in range(len(c_lines)):
            SC_Tokens = tokenizer.tokenize(c_lines[index])
            Tokens = [tokenizer.cls_token] + SC_Tokens + [tokenizer.eos_token]
            if len(Tokens)>=512:
                Tokens = Tokens[0:511]+[tokenizer.eos_token]
            tokens_id = tokenizer.convert_tokens_to_ids(Tokens)
            tokens_tensor = torch.tensor(tokens_id)[None, :]
            with torch.no_grad():
                try:
                    results = model(tokens_tensor)
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        print('WARNING: out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
            line_repr = results.last_hidden_state[0][0]
            torch.save(line_repr,pj_path+"\\all_line_codeBert\\"+str(index+1)+".pt")







if __name__ == "__main__":
    process_func()
    process_line()
#     reverse()


