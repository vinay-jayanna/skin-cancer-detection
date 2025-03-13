FROM python:3.11.2-slim

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install required dependencies
RUN pip install --no-cache-dir \
    torch torchvision pillow numpy pandas mlserver opencv-python-headless

# Copy model files and custom runtime
COPY resnet50_skin_cancer_model.pth /app/
COPY settings.json /app/
COPY custom_runtime.py /app/
COPY HAM10000_metadata.csv /app/

# Expose MLServer port
EXPOSE 8080

# Set MLServer as the entrypoint
CMD ["mlserver", "start", "."]
