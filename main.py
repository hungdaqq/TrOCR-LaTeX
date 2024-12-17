from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io
import time
import torch
import base64
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

app = FastAPI(title="OCR API", description="API for OCR Prediction using TrOCR")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = TrOCRProcessor.from_pretrained("./trocr-base-finetuned-math-captions")
model = VisionEncoderDecoderModel.from_pretrained(
    "./trocr-base-finetuned-math-captions"
).to(device)


class OcrRequest(BaseModel):
    image_base64: str


class OcrResponse(BaseModel):
    latex: str
    inference_time: float


@app.post("/api/ocr", response_model=OcrResponse)
async def ocr_predict(request: OcrRequest):
    """
    OCR prediction on base64 image
    """
    start_time = time.time()

    try:
        # Decode the base64 image
        image_data = base64.b64decode(request.image_base64)
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        output = model.generate(pixel_values, max_new_tokens=256)
        prediction = processor.decode(output[0], skip_special_tokens=True).replace(
            "\\ ", "\\"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during OCR processing: {str(e)}"
        )

    finally:
        end_time = time.time()
        inference_time = end_time - start_time
        print(f"TrOCR inference time: {inference_time:.2f} seconds")

    return JSONResponse(
        content={"latex": prediction, "inference_time": inference_time},
        media_type="application/json",
        status_code=200,
    )


# Use uvicorn to host the app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
