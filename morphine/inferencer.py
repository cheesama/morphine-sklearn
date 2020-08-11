from fastapi import FastAPI

import dill

app = FastAPI()
is_ready = False

#load intent, entity model
intent_model = None
entity_model = None
is_ready = True

#endpoints
@app.get("/"):
async def health():
    if is_ready:
        output = {'code': 200}
    else:
        output = {'code': 500}
    return output

@app.post("/predict/intent"):
async def predict_intent(text: str):
    return {}

@app.post("/predict/entity"):
async def predict_entity(text: str):
    return {}

 
