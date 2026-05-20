from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pandas as pd
import numpy as np
import pickle
import os

# Esto le dice a Python: "Busca la carpeta donde está este archivo app.py y úsala como base"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ahora unimos esa ruta con la carpeta models
model_path = os.path.join(BASE_DIR, "models", "modelec.pickle")
scaler_path = os.path.join(BASE_DIR, "models", "scalerec.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# Crear nuestra API con FastAPI
app = FastAPI()

# Carpeta static para enlazar tus estilos CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


# Ruta principal para cargar el formulario index.html
@app.get("/", response_class=HTMLResponse)
def formulario(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "resultado": None, "mensaje": None},
    )


# Función para recibir los datos y realizar la predicción
@app.post("/predict", response_class=HTMLResponse)
# Acoplamos la función con las 4 variables exactas de tu formulario HTML
def predecir(
    request: Request,
    age: float = Form(...),
    trestbps: float = Form(...),
    chol: float = Form(...),
    thalach: float = Form(...),
):
    # Convertimos los datos clínicos ingresados a un arreglo de NumPy (vector de una fila)
    # Cambia tu np.array por esta tabla de Pandas:
    datos = pd.DataFrame([[age, trestbps, chol, thalach]], columns=['age', 'trestbps', 'chol', 'thalach'])
    
    # Escalamos los nuevos datos usando tu escalador (scalerec.pkl)
    datos_estandarizados = scaler.transform(datos.values)  
    
    # Realizamos la predicción con el modelo SVC usando los datos escalados
    prediccion = model.predict(datos_estandarizados)

    # Obtenemos el resultado numérico (0 o 1)
    resultado = int(prediccion[0])
    
    # Lógica de diagnóstico médico en lugar de las flores Iris
    mensaje = ""
    if resultado == 0:
        mensaje = "El paciente no presenta indicios de enfermedad cardíaca (Bajo Riesgo)."
    else:
        mensaje = "El paciente presenta indicios de enfermedad cardíaca (Alto Riesgo). Se recomienda acudir al médico."

    # Retornamos el resultado y el mensaje de vuelta a tu index.html
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "resultado": resultado, "mensaje": mensaje}
    )