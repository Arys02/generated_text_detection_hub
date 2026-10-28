source: https://github.com/avduarte333/Sem-Detect
commit: d97175a513fab1aa97b31cc9a06874bad79a21c0
copied: 2026-09-30
modifications: none (ne pas modifier : wrapper dans src/detector/sem_detect/)

statut: NON REDISTRIBUÉ. Le dépôt amont n'a pas de licence : son code n'est donc pas versionné ici (dépôt public).
chemins concernés (ignorés par git, voir .gitignore) :
  - src/detector/sem_detect/semdetect/                    (17 fichiers)
  - src/detector/sem_detect/models/lightgbm_best_model.pkl
intégrité : src/detector/sem_detect/semdetect.sha256 (18 empreintes SHA-256, versionné)
récupération (clone de l'amont au commit ci-dessus, copie dans l'arborescence, vérification) :
    python -m scripts.fetch_semdetect            # installe, refuse d'écraser un fichier local différent
    python -m scripts.fetch_semdetect --check    # vérifie l'installation, sans réseau
citation :
    duarte2026semdetect (doc/references.bib)
    Duarte, Tufts, Oke, Fang, Oliveira, Li. Sem-Detect: Semantic Level Detection of AI Generated Peer-Reviews. ICML 2026 (PMLR 306).
