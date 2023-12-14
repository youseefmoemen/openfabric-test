import os
import warnings
from typing import Dict

from openfabric_pysdk.utility import SchemaUtil

from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import Ray, State
from openfabric_pysdk.loader import ConfigClass

import logging
from transformers import AutoTokenizer, AutoModelForTextEncoding, RagRetriever, AutoModelWithLMHead
import torch.nn as nn
import torch



class ContextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-nq")
        self.retriever = RagRetriever.from_pretrained(
            "facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True
            )
        self.question_encoder = AutoModelForTextEncoding.from_pretrained('bert-base-uncased').to(self.device)
    def forward(self, question: str) -> str:
        logging.info('Generating Context')
        input_ids = self.tokenizer(question, return_tensors="pt")['input_ids'].to(self.device)
        question_hidden_states = self.question_encoder(input_ids)['last_hidden_state'].squeeze(0)
        docs_dict = self.retriever(input_ids.cpu().numpy().reshape(-1), question_hidden_states.detach().cpu().numpy(), n_docs=1,return_tensors="pt")
        context_tokens = docs_dict['context_input_ids']
        context = self.tokenizer.batch_decode(context_tokens, skip_special_tokens=True)
        context = ' '.join(context)[:512]
        return context

class QAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')
        self.model = AutoModelWithLMHead.from_pretrained("google/flan-t5-large", device_map="auto", torch_dtype=torch.float16)

    def forward(self, question: str, context: str):
        logging.info('Generating Answer')
        prompt =  context + ' based on the previous information answer the following question: ' + question
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs.to('cuda'))
        answer = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return answer



############################################################
# Callback function called on update config
############################################################
def config(configuration: Dict[str, ConfigClass], state: State):
    pass


def model_loading(): 
    model_a = ContextModel()
    model_b = QAModel()
    return model_a, model_b

# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: Ray, state: State) -> SimpleText:
    logging.info('Started Loading Models')
    context_model, qa_model = model_loading()
    logging.info('Models Loaded')
    output = []
    logging.info(request.text)
    for text in request.text:
        with torch.no_grad():
            context = context_model(text)
            answer = qa_model(text, context)
            logging.info(text)
            logging.info(answer)
        output.append(answer)
        torch.cuda.empty_cache()
    return SchemaUtil.create(SimpleText(), dict(text=output))

