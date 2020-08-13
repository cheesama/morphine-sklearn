from fastapi import FastAPI

from utils import tokenize, convert_ner_data_format, sent2features

import dill
import pycrfsuite

app = FastAPI()
is_ready = False

#load intent, entity model
intent_model = None
entity_model = None
with open('./morphine_intent_model.svc', 'rb') as f:
    intent_model = dill.load(f)
    print ('intent model load success')

entity_model = pycrfsuite.Tagger()
entity_model.open('./morphine_entity_model.crfsuite')
print ('entity model load success')

if intent_model and entity_model:
    is_ready = True

#endpoints
@app.get("/")
async def health():
    if is_ready:
        output = {'code': 200}
    else:
        output = {'code': 500}
    return output

@app.post("/predict/intent")
async def predict_intent(text: str):
    name = intent_model.predict([text])[0]
    confidence = intent_model.predict_proba([text])[0].max()

    return {'name': name, 'confidence': confidence, 'Classifier': 'morphine_intent_model.svc'}

@app.post("/predict/entity")
async def predict_entity(text: str):
    feature = sent2features(convert_ner_data_format(text))
    entities = entity_model.tag(feature)
    tokens = tokenize(text)

    result = []
    token_value = ''
    entity_value = ''
    for i, (token, entity) in enumerate(zip(tokens, entities)):
        if entity != 'O':
            if i < len(entities) - 1 and entities[i][2:] == entities[i+1][2:]:
                entity_value = entity.replace('B-','').replace('I-','')
                token_value += token
            else:
                result.append({'entity': entity_value, 'value': token_value})
                token_value = ''
                entity_value = ''

    return {'entities': result, 'Extractor': 'morphine_entity_model.crfsuite'}

 
