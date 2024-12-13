import grpc
from concurrent import futures
import ocr_service_pb2
import ocr_service_pb2_grpc
import io
import time

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

processor = TrOCRProcessor.from_pretrained("./trocr-base-finetuned-math-captions")
model = VisionEncoderDecoderModel.from_pretrained(
    "./trocr-base-finetuned-math-captions"
).to(device)


class OcrServiceServicer(ocr_service_pb2_grpc.OcrServiceServicer):
    def OcrPredict(self, request, context):
        start_time = time.time()

        try:
            # Load and preprocess the image
            image = Image.open(io.BytesIO(request.image)).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)

            # Generate OCR prediction
            output = model.generate(pixel_values, max_new_tokens=256)
            prediction = processor.decode(output[0], skip_special_tokens=True).replace(
                "\\ ", "\\"
            )

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error during OCR processing: {str(e)}")
            return ocr_service_pb2.OcrResponse()

        end_time = time.time()
        computation_time = end_time - start_time
        print(
            f"TrOCR inference time: {computation_time:.2f} seconds"
        )  # Log the computation time

        return ocr_service_pb2.OcrResponse(
            latex=prediction,
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ocr_service_pb2_grpc.add_OcrServiceServicer_to_server(OcrServiceServicer(), server)
    server.add_insecure_port("[::]:50051")
    print("Server is running on port 50051...")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
