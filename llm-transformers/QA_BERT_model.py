from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import torch

def text_embeddings(question: str, context: str):
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    model = BertForQuestionAnswering.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    encoding = tokenizer.encode_plus(text=question, text_pair=context)
    print(encoding)

    inputs = encoding['input_ids']
    sentence_embedding = encoding['token_type_ids']
    tokens = tokenizer.convert_ids_to_tokens(inputs)

    output = model(input_ids = torch.tensor([inputs]), token_type_ids = torch.tensor([sentence_embedding]))

    start_index = torch.argmax(output.start_logits)
    end_index = torch.argmax(output.end_logits)

    answer = ' '.join(tokens[start_index:end_index+1])
    print(answer)     

