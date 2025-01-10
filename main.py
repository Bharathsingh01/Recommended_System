import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from recommender import  recommend_system

app = FastAPI()

with open("recommend.pkl", "rb") as file:
    saved_recommendations = pickle.load(file)  # This will now work

class RecommendationRequest(BaseModel):
    user_id: int | None = None
    item_name: str

@app.post("/recommend/")
def get_recommendations(request: RecommendationRequest):
    train_data = pd.read_csv("/Users/bharathsingh/Downloads/datas_1.csv")  # Load your dataset here
    recommendations = recommend_system(request.user_id, request.item_name, train_data)
    return {"recommendations": recommendations}
if __name__ == "__main__":
    uvicorn.run(app,host ='127.0.0.1', port=5000)