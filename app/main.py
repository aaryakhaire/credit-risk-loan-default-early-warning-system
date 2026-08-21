from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import ApplicantData
from app.model import predict_risk

app = FastAPI(title="Credit Risk Early Warning System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(applicant: ApplicantData):
    return predict_risk(applicant.dict())