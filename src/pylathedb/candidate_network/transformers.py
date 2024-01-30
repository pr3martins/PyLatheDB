from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class BERTHandler:
    def __init__(self, model_name = None):
        self.model = None
        self.tokenizer = None
        if model_name is None:
            self.model_name = 'sentence_transformers/all-mpnet-base-v2'
        else:
            self.model_name = model_name

    def load_model(self):
        print(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        return self.model

    def encode(self,sentences):
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0] #First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        #CLS Pooling - Take output from first token
        def cls_pooling(model_output):
            return model_output.last_hidden_state[:,0]

        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        
        if self.model_name != 'multi-qa-mpnet-base-dot-v1':
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            # Perform pooling
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        else:
            # Compute token embeddings
            with torch.no_grad():
                model_output = model(**encoded_input, return_dict=True)
            # Perform pooling
            embeddings = cls_pooling(model_output)

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings