from fastapi import FastAPI

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
    return {}

@app.post("/predict/entity")
async def predict_entity(text: str):
    return {}

 
