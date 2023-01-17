import os

if __name__ == '__main__':
    CRF_MODEL_PATH = './model/T5-BiLSTM-CRF'
    BERT_PATH = './model/ProtT5'
    model_T5_list = os.listdir(BERT_PATH)
    if "config.json" not in model_T5_list:
        os.system("wget  https://huggingface.co/Rostlab/prot_t5_xl_uniref50/resolve/main/config.json -O ./model/ProtT5/config.json")
    if "pytorch_model.bin" not in model_T5_list:
        os.system("wget https://huggingface.co/Rostlab/prot_t5_xl_uniref50/resolve/main/pytorch_model_723k.bin -O ./model/ProtT5/pytorch_model.bin")
    if "special_tokens_map.json" not in model_T5_list:
        os.system("wget https://huggingface.co/Rostlab/prot_t5_xl_uniref50/resolve/main/special_tokens_map.json -O ./model/ProtT5/special_tokens_map.json")
    if "spiece.model" not in model_T5_list:
        os.system("wget https://huggingface.co/Rostlab/prot_t5_xl_uniref50/resolve/main/spiece.model -O ./model/ProtT5/spiece.model")
    if "tokenizer_config.json" not in model_T5_list:
        os.system("wget https://huggingface.co/Rostlab/prot_t5_xl_uniref50/resolve/main/tokenizer_config.json -O ./model/ProtT5/tokenizer_config.json")

    model_graph_list = os.listdir(CRF_MODEL_PATH)
    if "precision:0.8473289966489257,recall:0.8473289966489257,f1:0.8473289966489257.pt" not in model_graph_list:
        os.system("wget https://inner.wei-group.net/result/precision:0.8473289966489257,recall:0.8473289966489257,f1:0.8473289966489257.pt -O ./model/T5-BiLSTM-CRF/precision:0.8473289966489257,recall:0.8473289966489257,f1:0.8473289966489257.pt")