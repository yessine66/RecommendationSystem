from fastapi import BackgroundTasks, FastAPI
import uvicorn
import os
import logging
import numpy as np

from modeling import load_model, predict, run_training



#load enviroment variables
#port = int(os.environ["PORT"])
port = 80
print(port)

app = FastAPI()

if os.listdir("../models") != 0 :
    print("model exist *************************f")
    model = load_model("etafakna.joblib")
    print("result *****",predict([45,0,0,0,0,0], model))
else :
    run_training()
    

@app.get("/hello")
async def say_hello():
    return {"message": "hello world!"}


@app.get("/predict/{userx}")
async def predict_contract(userx):
    input=np.fromstring(userx,sep=',')
    result = predict(input, model)
    return{
        "userx": result.tolist()
    }
 