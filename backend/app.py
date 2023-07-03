import jwt
from fastapi import FastAPI, HTTPException
from fastapi import Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
import psycopg2

from model import get_segmentator, get_segments
from starlette.responses import JSONResponse, Response
from fastapi.responses import StreamingResponse
import numpy as np
import asyncio
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import cv2
import json
import io
from dotenv import load_dotenv
import os
from huggingface_hub import hf_hub_download
import joblib


load_dotenv()
app = FastAPI()

# Connection information
host = "dpg-chbqtq2k728tp9fs6cjg-a.oregon-postgres.render.com"
port = "5432"
database = "database_aq6s"
user = "aayush"
password = "gOSe9IhyVujV4z1dzQ2cvDsyHhbk2uIk"

# Construct the DSN string
dsn = f"postgres://{user}:{password}@{host}:{port}/{database}"

# Establish the connection
connection = psycopg2.connect(dsn)


REPO_ID = "https://huggingface.co/aadh-goa/brainmri"
FILENAME = "model.h5"

model = joblib.load(
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
)

# security settings
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = "your-secret-key"

# CORS settings
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3002",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# user registration


@app.post("/register")
async def register(username: str, email: str, password: str, confirm_password: str):
    if password != confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    hashed_password = pwd_context.hash(password)
    try:
        cur.execute(
            "INSERT INTO users (username,email,password) VALUES (%s,%s, %s)",
            (username, email, hashed_password),
        )
        conn.commit()
        return {"message": "User registered successfully"}
    except:
        raise HTTPException(status_code=400, detail="Username already exists")


# user login
@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    username = form_data.username
    password = form_data.password

    cur.execute("SELECT password FROM users WHERE username = %s", (username,))
    result = cur.fetchone()
    if not result:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    hashed_password = result[0]
    if not pwd_context.verify(password, hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    # generate JWT token for authentication
    access_token = jwt.encode({"sub": username}, SECRET_KEY)
    return {"access_token": access_token, "token_type": "bearer"}


# protected endpoint


@app.get("/protected")
async def protected(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        username = payload["sub"]
        return {"message": f"Hello, {username}!"}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


model = get_segmentator()

application = FastAPI(
    title="DeepLabV3 image segmentation",
    description="""Obtain semantic segmentation maps of the image in input via DeepLabV3 implemented in PyTorch.
                           Visit this URL at port 8501 for the streamlit interface.""",
    version="0.1.0",
)


@app.post("/unet")
async def get_segmentation_map(image: UploadFile = File(...)):
    file_bytes = await image.read()
    image_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    segmented_image = get_segments(model, image)

    segmented_image = cv2.cvtColor(
        np.array((segmented_image * 255), dtype=np.uint8), cv2.COLOR_BGR2GRAY
    )
    print(segmented_image)
    img_bytes = cv2.imencode(".png", segmented_image)[1].tobytes()

    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend:app", host="localhost", port="8001")
