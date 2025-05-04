from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import opennsfw2 as n2
import aiohttp
import asyncio
import tempfile
import os

app = FastAPI()


class ImageURLs(BaseModel):
    urls: List[str]


async def download_image(session, url, folder):
    try:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download: {url}")

            file_path = os.path.join(folder, os.path.basename(url))
            with open(file_path, "wb") as f:
                f.write(await response.read())
            return file_path
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/nsfw-check/")
async def check_nsfw_images(data: ImageURLs):
    if not data.urls:
        raise HTTPException(status_code=400, detail="No image URLs provided.")

    with tempfile.TemporaryDirectory() as tmpdir:
        async with aiohttp.ClientSession() as session:
            tasks = [download_image(session, url, tmpdir) for url in data.urls]
            local_paths = await asyncio.gather(*tasks)

        try:
            probabilities = n2.predict_images(local_paths)
            is_safe = not any(prob > 0.3 for prob in probabilities)
            return {
                "is_safe": is_safe,
                "nsfw_probabilities": probabilities
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
