import base64
import io
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput
from torchvision.models import resnet50

class SkinCancerModel(MLModel):
    async def load(self) -> bool:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet50(weights=None)  # Load ResNet50 without pre-trained weights
        
        # Load metadata to retrieve class mappings
        metadata_csv_path = "/app/HAM10000_metadata.csv"
        df = pd.read_csv(metadata_csv_path)
        self.label_mapping = {label: idx for idx, label in enumerate(df['dx'].unique())}
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}  # Reverse mapping
        
        # Modify final layer to match number of classes
        num_classes = len(self.label_mapping)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        
        # Load trained weights
        model_path = self.settings.parameters.uri
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        
        # Define image transform
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        return True

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        images = []
        
        for input_data in payload.inputs:
            if input_data.name == "image_base64":
                # Decode base64 image
                img_bytes = base64.b64decode(input_data.data[0])
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                image = self.transform(image).unsqueeze(0).to(self.device)
                images.append(image)
        
        # Convert list of images to a single tensor
        input_tensor = torch.cat(images, dim=0)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted = torch.max(outputs, 1)  # Get class index
        
        predicted_classes = [self.reverse_label_mapping[pred.item()] for pred in predicted]
        print(predicted_classes)

        # Dictionary mapping classes to descriptions
        class_descriptions = {
            "nv": "Nevus (NV) is commonly known as a mole, a benign growth of melanocytes, the pigment-producing cells in the skin. They appear as small, round, brown, or black spots and are usually harmless. However, changes in size, shape, or color can indicate a potential risk of melanoma. Regular monitoring is advised, especially for individuals with multiple or atypical moles.",
            "mel": "Melanoma (MEL) is the most dangerous form of skin cancer, originating from melanocytes and often linked to excessive UV exposure. It can appear as an irregularly shaped, dark-colored lesion that may evolve in size, shape, or texture over time. Early detection is crucial, as melanoma can spread rapidly to other parts of the body. Treatment typically involves surgical removal, and advanced cases may require immunotherapy or chemotherapy.",
            "bkl": "Benign Keratosis (BKL) includes non-cancerous skin lesions such as seborrheic keratosis and solar lentigines, often caused by aging and sun exposure. These growths are usually rough, scaly, and can be light brown, black, or yellowish. While they are harmless, they may resemble more serious conditions like melanoma, requiring professional diagnosis. Removal options include cryotherapy, laser treatment, or minor surgical procedures.",
            "bcc": "Basal Cell Carcinoma (BCC) is the most common type of skin cancer, originating in the basal cells of the epidermis due to prolonged UV exposure. It often presents as a pearly or waxy bump, sometimes with visible blood vessels or ulceration. While it rarely spreads to other parts of the body, untreated cases can cause significant local tissue damage. Treatment options include surgical excision, radiation therapy, and topical medications.",
            "akiec": "Actinic Keratosis (AKIEC) is a precancerous skin condition resulting from prolonged sun exposure, often appearing as rough, scaly patches on sun-exposed areas like the face, scalp, and hands. While not immediately cancerous, these lesions can develop into squamous cell carcinoma if left untreated. Early intervention with cryotherapy, laser therapy, or topical medications can prevent progression. Regular skin checks are essential for those at high risk due to UV exposure.",
            "df": "Dermatofibroma (DF) is a benign skin lesion that often appears as a firm, reddish-brown bump on the skin, commonly found on the lower legs. It is typically harmless but may be confused with malignant lesions. Dermatofibromas are composed of fibrous tissue and may cause mild discomfort when pressed. Treatment is generally not required unless the lesion becomes bothersome.",
            "vasc": "Vascular Lesions (VASC) are a group of skin conditions that involve blood vessels, including angiomas and hemangiomas. They often appear as small, red or purple raised bumps on the skin. Most vascular lesions are benign and do not require treatment, but some may be removed for cosmetic reasons or if they cause discomfort."
        }

        descriptions = [class_descriptions.get(pred_class, "No description available") for pred_class in predicted_classes]

        return InferenceResponse(
            model_name=self.name,
            outputs=[
                ResponseOutput(
                    name="predictions",
                    shape=[len(predicted_classes)],
                    datatype="BYTES",
                    data=predicted_classes
                ),
                ResponseOutput(
                    name="descriptions",
                    shape=[len(descriptions)],
                    datatype="BYTES",
                    data=descriptions
                )
            ]
        )
