FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set working directory
WORKDIR /app

# Copy the application files into the container
COPY . .

# Update package list and install python3, pip, and other dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* 

# Install Python dependencies from requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

RUN nvidia-smi
# Generate Python bindings for gRPC service
# RUN python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ocr_service.proto

# Expose the required port
EXPOSE 50051

# Set the command to run the server
CMD ["python3", "server.py"]
