from vipas import model, exceptions
import os
import base64
import json

# Set authentication token
os.environ["VPS_AUTH_TOKEN"] = "vps-5bOyAeBMlopouyMBPaDv"
model_client = model.ModelClient()

try:
    # Corrected file path (changed .png to .jpg)
    image_path = "curated_images/ISIC_0026645.jpg"

    # Verify file existence
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    # Read the image file and encode it in base64
    with open(image_path, "rb") as image_file:
        base_64_string = base64.b64encode(image_file.read()).decode("utf-8")

    # Create input JSON body
    input_body = {
        "inputs": [
            {
                "name": "image_base64",
                "datatype": "BYTES",
                "shape": [1],
                "data": [base_64_string]
            }
        ]
    }

    # Send prediction request
    response = model_client.predict(model_id="model id", input_data=json.dumps(input_body))

    # Print base64 encoded output
    if response:
        print(response.outputs[0].data[0])

except FileNotFoundError as e:
    print(f"Error: {e}")
except exceptions.ClientException as e:
    print(f"Client Exception: {e}")
except Exception as e:
    print(f"Unexpected Error: {e}")
