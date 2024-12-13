import grpc
import ocr_service_pb2
import ocr_service_pb2_grpc

def send_image(image_path):
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = ocr_service_pb2_grpc.OcrServiceStub(channel)
        with open(image_path, "rb") as f:
            image = f.read()
        response = stub.OcrPredict(
            ocr_service_pb2.OcrRequest(image=image)
        )
        print(f"Server Response: {response.latex}")

if __name__ == "__main__":
    send_image("./handwritten_val/q5.png")
