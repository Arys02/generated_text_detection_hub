#evalution/eval_engine.py
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
import logging
from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, model, dataloader, device: str, type:str, save_dir: str = None):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.type = type
        #self.save_dir = save_dir  # Pour sauvegarder les logits bruts (optionnel)

    def run(self) -> Dict[str, float]:
        """Exécute la passe d'évaluation complète."""
        #self.model.eval()
        #self.model.to(self.device)

        # Conteneurs pour accumuler les résultats sur tout le dataset
        all_logits_tt = []
        all_labels_tt = []
        all_id_tt = []
        all_txt_tt = []
        #all_logits_aa = [] # Décommenter pour l'attribution
        #all_labels_aa = []

        logger.info(f"Démarrage inférence sur {self.device}...")

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Eval"):
                # 1. Transfert des données (Textes restents sur CPU généralement pour tokenisation)
                # texts = batch['content']
                # labels_tt = batch['label_tt'].to(self.device)
                # TODO switch between dataset more smart to do
                texts = batch['content']


                labels_tt = batch['label'].to(self.device)


                # 2. Forward Pass
                if self.type == "contextual":
                    outputs = self.model.predict(batch)
                else:
                    outputs = self.model.predict(texts)

                all_logits_tt.append(outputs)
                all_labels_tt.append(labels_tt.cpu().numpy())
                all_id_tt.append(batch['review-id'])
                all_txt_tt.append(batch['content'])

        # 4. Aggregation

        y_scores = np.concatenate(all_logits_tt).ravel()  # Flatten pour (N,)
        y_true = np.concatenate(all_labels_tt).ravel()
        ids = np.concatenate(all_id_tt).ravel().tolist()
        content = np.concatenate(all_txt_tt).ravel().tolist()



        #TODO add threshold
        # 5. Calcul Métriques
        logger.info("Calcul des métriques...")

        if hasattr(self.model.config, "threshold"):
            threshold = self.model.config.threshold
        else:
            threshold = 0.5
        metrics = compute_metrics(y_true, y_scores, threshold)

        self._log_results(metrics)
        return {**metrics, "ids":ids, "content":content}

    def _log_results(self, metrics: Dict[str, float]):
        """Affiche et sauvegarde les résultats."""
        logger.info("=== RÉSULTATS ===")
        for k, v in metrics.items():
            if not (isinstance(v, np.ndarray) or isinstance(v, list)):
                logger.info(f"{k.upper()}: {v:.4f}")

        # Ici on pourrait sauvegarder un JSON dans self.save_dir