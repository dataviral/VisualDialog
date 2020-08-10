import contractions
import re
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tf
from PIL import Image


from dialog.model import VisualDialog


transform = tf.Compose([
    tf.ToPILImage(),
    tf.Resize((224, 224)),
    tf.ToTensor(),
    tf.Normalize((0, 0, 0), (1, 1, 1)),
])

def fix_text(sentence):
    sentence = re.findall(r"[\w']+|[.,!?;]", contractions.fix(sentence.lower()))
    sentence = ["<sos>"] + sentence + ["<eos>"]
    return sentence

def tokenize_sent(sentence, mappings):
    sentence = fix_text(sentence)
    return [mappings["w2i"].get(i, mappings["w2i"]["<unk>"]) for i in sentence]

def untokenize_sent(model_output, mappings):
    s = []
    for word in model_output:
        if word == mappings["w2i"]["<pad>"] or word == mappings["w2i"]["<sos>"]:
            continue
        if word == mappings["w2i"]["<eos>"]:
            break
        s.append(word)
    return " ".join([mappings["i2w"][i] for i in s]).capitalize()

def get_model(model_path, mappings, device):
    feature_extractor = torchvision.models.resnet18(pretrained=True)
    for param in feature_extractor.parameters():
        param.requires_grad = False
    feature_extractor.fc = torch.nn.Linear(512, 512)

    feature_extractor = feature_extractor.to(device)

    model = VisualDialog(feature_extractor, len(mappings["w2i"]), device).to(device)
    model.load_state_dict(torch.load(model_path))
    return model

def predict_answer(question, img, mappings, model, reset):
    
    # Process question
    question = tokenize_sent(question, mappings)
    question = torch.LongTensor(question).unsqueeze(0)

    # Process Image
    img = torch.Tensor(img)
    img = transform(img)
    
    answer = model.predict(question, img, mappings, reset)
    return untokenize_sent(answer, mappings)