from fastapi import FastAPI, File
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import pickle
import warnings

import base64
from PIL import Image
import io

warnings.simplefilter(action='ignore', category=DeprecationWarning)

app = FastAPI()

# Definição dos tipos de dados
class PredictionResponse(BaseModel):
    prediction: float

class ImageRequest(BaseModel):
    image: str
    # Carregamento do Modelo de Machine Learning
def load_model():
    global xgb_model_carregado
    with open("./notebooks/RandomFo.pkl", "rb") as f:
        xgb_model_carregado = pickle.load(f)

# Inicialização da Aplicação
@app.on_event("startup")
async def startup_event():
    load_model()

# Definição do endpoint /predict que aceita as requisições via POST
# Esse endpoint irá receber a imagem em base64 e irá convertê-la para fazer inferência
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ImageRequest):
    # Processamento da Imagem
    img_bytes = base64.b64decode(request.image)
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((8, 8))
    img_array = np.array(img)

    # Converter a imagem para escala de cinza
    img_array = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])

    # Preparar imagem para o modelo
    img_array = img_array.reshape(1, -1)

    # Predição do Modelo de Machine Learning
    prediction = xgb_model_carregado.predict(img_array)
    return {"prediction": prediction}

# Endpoint de Healthcheck
@app.get("/healthcheck")
async def healthcheck():
    # retorna um objeto com um campo status com valor "ok" se a aplicação estiver funcionando corretamente
    return {'status': 'ok'}