import json
from pathlib import Path
import requests
import fitz  # pymupdf
import os
from openreview.api import OpenReviewClient as ClientV2
from openreview import Client as ClientV1, openreview

from src.config import CACHE_DIR

cache_dir = os.getenv('CACHE_DIR')
print(cache_dir)


def make_clients() -> tuple[ClientV1, ClientV2]:
    user = os.environ["OPENREVIEW_LOGIN"]
    pwd = os.environ["OPENREVIEW_MDP"]
    v1 = ClientV1(
        baseurl="https://api.openreview.net",
        username=user, password=pwd,
    )
    v2 = ClientV2(
        baseurl="https://api2.openreview.net",
        username=user, password=pwd,
    )
    return v1, v2

class clients_cls():
    def __init__(self):
        self.v1 = None
        self.v2 = None
    def get_clients(self):
        if self.v1 is None or self.v2 is None:
            self.v1, self.v2 = make_clients()
        return self.v1, self.v2

client_obj = clients_cls()

def get_forum_id_from_review(review_id: str) -> str:
    """
    Récupère le forum_id (ID du papier) à partir d'un review_id via l'API OpenReview.
    """
    # 1. Tentative sur l'API V1
    url_v1 = f"https://api.openreview.net/notes?id={review_id}"
    response_v1 = requests.get(url_v1, timeout=10).json()

    if "notes" in response_v1 and len(response_v1["notes"]) > 0:
        return response_v1["notes"][0].get("forum")

    # 2. Repli sur l'API V2 si la note n'est pas dans la V1
    url_v2 = f"https://api.openreview.net/v2/notes?id={review_id}"
    response_v2 = requests.get(url_v2, timeout=10).json()

    if "notes" in response_v2 and len(response_v2["notes"]) > 0:
        return response_v2["notes"][0].get("forum")

    return None


def fetch_pdf_from_review_id(review_id: str, output_dir: str = os.path.join(os.getenv('CACHE_DIR', '.'), "papers_pdf")):
    """
    Pipeline complet : review_id -> forum_id -> téléchargement du PDF.
    """
    # print(f"Extraction du forum pour la review {review_id}...")
    forum_id = get_forum_id_from_review(review_id)

    if not forum_id:
        # print(f"Échec : Impossible de trouver le forum_id pour la review {review_id}.")
        return False

    # print(f"Forum trouvé : {forum_id}. Téléchargement du PDF...")

    # Réutilisation de la logique de l'endpoint PDF
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{forum_id}.pdf")
    pdf_url = f"https://openreview.net/pdf?id={forum_id}"

    if os.path.exists(file_path):
        return file_path
    try:
        response = requests.get(pdf_url, stream=True, timeout=10)
        response.raise_for_status()

        if 'application/pdf' not in response.headers.get('Content-Type', ''):
            print("Échec : Le contenu renvoyé n'est pas un PDF valide.")
            return None

        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Succès : {file_path}")
        return file_path

    except requests.exceptions.RequestException as e:
        print(f"Erreur HTTP lors du téléchargement : {e}")
        return None


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Extrait le texte brut d'un PDF de papier académique.
    Utilise pymupdf avec sort=True pour respecter l'ordre de lecture
    sur les layouts multi-colonnes.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        # sort=True ordonne les blocs par position (haut→bas, gauche→droite)
        pages.append(page.get_text("text", sort=True))
    doc.close()
    return "\n".join(pages)


def get_forum_id_from_review_v2(review_id: str) -> str | None:
    cache_mapping_path = CACHE_DIR / "review_paper_mapping.json"
    cache : dict = {}
    if os.path.exists(cache_mapping_path):
        with open(cache_mapping_path, "r") as f:
            cache = json.load(f)
        if review_id in cache:
            return cache[review_id]

    V1, V2 = client_obj.get_clients()
    for client in (V1, V2):
        try:
            note = client.get_note(review_id)
            if note and note.forum:
                cache[review_id] = note.forum
                with cache_mapping_path.open("w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)
                return note.forum
        except openreview.OpenReviewException:
            continue
    return None


def fetch_pdf_from_review_id_v2(
        review_id: str,
        output_dir: str = os.path.join(os.getenv("CACHE_DIR", "."), "papers_pdf"),
) -> str | None:
    V1, V2 = client_obj.get_clients()
    forum_id = get_forum_id_from_review_v2(review_id)
    if not forum_id:
        return None

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{forum_id}.pdf")
    if os.path.exists(file_path):
        return file_path

    for client in (V2, V1):  # V2 d'abord, plus de papiers récents
        try:
            pdf_bytes = client.get_pdf(id=forum_id)
            with open(file_path, "wb") as f:
                f.write(pdf_bytes)
            return file_path
        except openreview.OpenReviewException:
            continue
    return None
