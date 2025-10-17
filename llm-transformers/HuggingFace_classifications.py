from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def classify_sentence_using_torch(sentence: str, model_directory: str):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    input_ids_pt = tokenizer(sentence, return_tensors ="pt")
    print(input_ids_pt)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    with torch.no_grad():
        logits = model(**input_ids_pt).logits

    predicted_class_id = logits.argmax().item()
    model.config.id2label[predicted_class_id]

    # Save the pretrained model
    tokenizer.save_pretrained(model_directory)
    model.save_pretrained(model_directory)

# To load the model for usage
def load_pretrained_mode(model_directory: str):
    my_tokenizer = AutoTokenizer.from_pretrained(model_directory)
    my_model = AutoModelForSequenceClassification.from_pretrained(model_directory)

