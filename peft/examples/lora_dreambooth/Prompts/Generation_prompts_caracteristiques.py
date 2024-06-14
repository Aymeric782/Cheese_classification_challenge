import pandas as pd
import random
import math
import csv


contexts = [
    "with a close-up shot focusing on the texture of the cheese",
    "placed in the center of the frame with soft lighting to accentuate its texture",
    "surrounded by assorted crackers and fruits, with the cheese as the focal point",
    "sitting on a rustic wooden cheese board, perfectly lit to highlight its colors",
    "captured from a bird's eye view, showcasing the intricate patterns and shapes of the cheese",
    "featured against a dark background to make its vibrant color pop",
    "photographed on a marble slab with elegant garnishes, drawing attention to its appeal",
    "presented on a minimalist plate with subtle props, allowing the cheese to take center stage",
    "set against a backdrop of lush greenery, evoking a sense of freshness and indulgence",
    "featured in a close-up shot with selective focus, emphasizing its texture",
    "positioned against a rustic backdrop, evoking a cozy and inviting atmosphere",
    "captured in soft, diffused light to enhance its appearance",
    "placed on a vintage serving platter, adding a touch of nostalgia to the scene",
    "surrounded by complementary ingredients, creating a visually appealing composition",
    "photographed with shallow depth of field to create a dreamy and inviting ambiance",
    "served on a modern ceramic plate, juxtaposing tradition with contemporary aesthetics",
    "captured in natural sunlight, highlighting its organic and wholesome qualities",
    "presented on a rustic wooden table, invoking feelings of warmth and comfort",
    "featured in a minimalist setting, allowing its natural beauty to shine through",
    "photographed in dramatic lighting, adding a sense of intrigue and sophistication"
]


# Lire le fichier Excel
file_path = "cheese_challenge.xlsx"
df = pd.read_excel(file_path, header=None)

# La première ligne contient les noms des fromages, donc on la définit comme noms de colonnes
df.columns = df.iloc[0]
df = df[1:]

# Transposer le DataFrame pour avoir les fromages comme lignes
df = df.transpose()

# Définir les noms des colonnes
df.columns = df.iloc[0]
df = df[1:]

# Convertir les noms de colonnes en chaînes de caractères et les nettoyer
df.columns = [str(col).strip().replace(' ', '').replace("'", "") for col in df.columns]


# Fonction pour sélectionner aléatoirement un adjectif ou un synonyme s'il y en a plusieurs
def get_random_adjective(adjectives):
    if pd.isna(adjectives):
        return ''
    options = adjectives.split(',')
    return random.choice(options).strip()

# Fonction pour générer un prompt pour un fromage
def generate_prompt(row):
    characteristics = {
        'Matièrepremiere': f"made from {get_random_adjective(row['Matièrepremiere'])} milk" if 'Matièrepremiere' in row and not pd.isna(row['Matièrepremiere']) else '',
        'Paysdorigine': f"from {row['Paysdorigine']}" if 'Paysdorigine' in row and not pd.isna(row['Paysdorigine']) else '',
        'Région': f"from the {row['Région']} region" if 'Région' in row and not pd.isna(row['Région']) else '',
        'Famille': f"{row['Famille']} cheese" if 'Famille' in row and not pd.isna(row['Famille']) else '',
        'Type': f"{row['Type']} cheese" if 'Type' in row and not pd.isna(row['Type']) else '',
        'Texture': f"with a {get_random_adjective(row['Texture'])} texture" if 'Texture' in row and not pd.isna(row['Texture']) else '',
        'Croûte': f"with a {get_random_adjective(row['Croûte'])} rind" if 'Croûte' in row and not pd.isna(row['Croûte']) else '',
        'Couleur': f"with a {get_random_adjective(row['Couleur'])} color" if 'Couleur' in row and not pd.isna(row['Couleur']) else '',
        'Saveur': f"with a {get_random_adjective(row['Saveur'])} flavor" if 'Saveur' in row and not pd.isna(row['Saveur']) else '',
        'Arôme': f"with a {get_random_adjective(row['Arôme'])} aroma" if 'Arôme' in row and not pd.isna(row['Arôme']) else '',
        'Synonymes': f"also known as {get_random_adjective(row['Synonymes'])}" if 'Synonymes' in row and not pd.isna(row['Synonymes']) else ''
    }

    # Filtrer les caractéristiques vides
    filtered_characteristics = {k: v for k, v in characteristics.items() if v}
    
    prompts = []
    for _ in range(400):
        # Sélectionner un nombre aléatoire de caractéristiques
        num_characteristics = random.randint(1, len(filtered_characteristics))
        selected_characteristics = random.sample(list(filtered_characteristics.values()), num_characteristics)

        # Construire le prompt
        prompt = f"an image of {row.name} cheese"
        for characteristic in selected_characteristics:
            prompt += f", {characteristic}"
        prompts.append(prompt+" "+random.choice(contexts))
    
    return prompts

for index, row in df.iterrows():
    prompts = generate_prompt(row)
    file_name = f"{row.name.strip().upper()}.csv"
    with open(file_name, 'w', newline='', encoding='utf-8') as fichier_csv:
        writer = csv.writer(fichier_csv)
        for prompt in prompts:
            writer.writerow([prompt])