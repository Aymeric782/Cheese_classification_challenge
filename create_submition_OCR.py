import hydra
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd
import torch

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes, ComputerVisionOcrErrorException
from msrest.authentication import CognitiveServicesCredentials
import spacy
import time
from fuzzywuzzy import fuzz


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ocr(image_path):
    # Charger le modèle de langue en anglais
    nlp = spacy.load("en_core_web_md")

    # Configuration Azure
    azure_endpoint = "https://testocrhrousseau.cognitiveservices.azure.com/"
    azure_key = "46d0903f287c4c00b9f7e39c65b20772"

    # Authentification avec la clé d'abonnement
    credentials = CognitiveServicesCredentials(azure_key)
    
    # Initialisation du client Computer Vision
    computervision_client = ComputerVisionClient(azure_endpoint, credentials)
    
    # Ouverture de l'image
    with open(image_path, "rb") as image_stream:

        # Appel de l'API Computer Vision pour l'OCR (Read API)
        read_response = computervision_client.read_in_stream(image_stream, raw=True)
        read_operation_location = read_response.headers["Operation-Location"]
        operation_id = read_operation_location.split("/")[-1]
        
        # Attente du résultat
        while True:
            read_result = computervision_client.get_read_result(operation_id)
            if read_result.status not in ['notStarted', 'running']:
                break
            time.sleep(1)
            
        # Récupération du texte détecté
        ocr_texts = []
        if read_result.status == OperationStatusCodes.succeeded:
            for page in read_result.analyze_result.read_results:
                for line in page.lines:
                    ocr_texts.extend([word.text for word in line.words])
        
    # Liste des noms de fromages possibles
    fromages_possibles = [
        "brie de melun", "camembert", "epoisses", "fourme d'ambert", "fourme", "ambert", "bûche", "raclette",
        "morbierr", "saint-nectaire", "pouligny saint- pierre", "roquefort", "comte", "pecorino", "neufchatel", "cheddar", "buchette de chevre", "parmesan",
        "saint- felicien", "mont d'or", "stilton", "scarmoza", "cabecou", "beaufort",
        "munster", "chabichou", "tomme de vache", "reblochon", "emmental", "feta",
        "ossau- iraty", "mimolette", "maroilles", "gruyere", "motheais", "vacherin",
        "mozzarella", "tete de moines", "fromage frais", "chevre"
    ]

    # Nettoyage des textes OCR
    cleaned_texts = [text.lower().replace('.', '').replace(',', '') for text in ocr_texts]
    
    # Identifier le fromage dans le texte extrait
    fromage_identifie = None
    similarity_max = 0
    for fromage in fromages_possibles:
        for text in cleaned_texts:
            similarity = fuzz.ratio(fromage.lower(), text.lower())
            if similarity > similarity_max:
                similarity_max = similarity
                fromage_identifie = fromage

    # Vérification des combinaisons de mots
    for i in range(len(cleaned_texts) - 1):
        combined_text = cleaned_texts[i] + ' ' + cleaned_texts[i + 1]
        for fromage in fromages_possibles:
            similarity = fuzz.ratio(fromage.lower(), combined_text.lower())
            if similarity > similarity_max:
                similarity_max = similarity
                fromage_identifie = fromage

    if fromage_identifie=="fourme" or fromage_identifie=="ambert":
        fromage_identifie="fourme d'ambert"
    if fromage_identifie=="bûche":
        fromage_identifie="buchette de chevre"
    if fromage_identifie and similarity_max >= 60:
        return True, fromage_identifie
    else:
        return False, ""



class TestDataset(Dataset):
    def __init__(self, test_dataset_path, test_transform):
        self.test_dataset_path = test_dataset_path
        self.test_transform = test_transform
        images_list = os.listdir(self.test_dataset_path)
        # filter out non-image files
        self.images_list = [image for image in images_list if image.endswith(".jpg")]

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image_path = os.path.join(self.test_dataset_path, image_name)
        image = Image.open(image_path)
        image = self.test_transform(image)
        return image, os.path.splitext(image_name)[0]

    def __len__(self):
        return len(self.images_list)


@hydra.main(config_path="configs/train", config_name="config_dreambooth")
def create_submission(cfg):
    test_loader = DataLoader(
        TestDataset(
            cfg.dataset.test_path, hydra.utils.instantiate(cfg.dataset.test_transform)
        ),
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
    )
    # Load model and checkpoint
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    checkpoint = torch.load(cfg.checkpoint_path)
    print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
    model.load_state_dict(checkpoint)
    class_names = sorted(os.listdir(cfg.dataset.train_path))

    # Create submission.csv
    submission = pd.DataFrame(columns=["id", "label"])

    for i, batch in enumerate(test_loader):
        images, image_names = batch
        images = images.to(device)
        preds = model(images)
        preds = preds.argmax(1)
        preds = [class_names[pred] for pred in preds.cpu().numpy()]

        #intervention de l'OCR
        for k in range(len(image_names)):
            b, name = ocr(cfg.dataset.test_path + "/" + image_names[k] + ".jpg")
            if b:
                preds[k] = name.upper()
        
        submission = pd.concat(
            [
                submission,
                pd.DataFrame({"id": image_names, "label": preds}),
            ]
        )
    submission.to_csv(f"{cfg.root_dir}/submission_dreambooth_OCR_V2.csv", index=False)


if __name__ == "__main__":
    create_submission()
