# Reconnaissance d'expressions faciales avec un CNN

Ce projet consiste à construire un réseau de neurones convolutif (CNN) avec Keras, capable de reconnaître l'expression d'un visage parmi 7 émotions. Le classifieur obtenu est ensuite branché sur un détecteur de visages YOLO pour analyser plusieurs visages dans une image, puis dans une vidéo.

Tout le travail et les détails le concernant se trouvent dans le notebook [emotion_classifier_cnn.ipynb](emotion_classifier_cnn.ipynb), prévu pour être exécuté sur Google Colab.

## Contexte

### Dataset

- **Source** : [RAF-DB sur Kaggle](https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset/data) (Shan Li, Weihong Deng)
- **7 classes** : Surprise, Peur, Dégoût, Joie, Tristesse, Colère, Neutre
- **Format** : images couleur RGB de 100×100 pixels (.jpg)
- **Effectif total** : 15 339 images, avec un fort déséquilibre entre les classes (5 957 images de Joie contre 355 de Peur)
- **Split** : 80 % train (12 271), 10 % validation (1 534), 10 % test (1 534), stratifié par classe

| Classe    | Effectif |
| :-------- | -------: |
| Surprise  |    1 619 |
| Peur      |      355 |
| Dégoût    |      877 |
| Joie      |    5 957 |
| Tristesse |    2 460 |
| Colère    |      867 |
| Neutre    |    3 204 |

### Démarche

1. **Exploration et préparation des données** : normalisation des pixels dans [0, 1], labels ramenés à 0–6, pipeline `tf.data`.
2. **Modèle de référence** : un MLP simple, qui sert de point de comparaison.
3. **CNN** : 3 blocs convolutifs.
4. **Boucle d'entraînement**, avec les mêmes paramètres pour tous les modèles :
   - optimiseur Adam, learning rate 0.001 ;
   - loss `sparse_categorical_crossentropy` ;
   - `class_weight="balanced"` pour compenser le déséquilibre des classes ;
   - batch size 64, 30 époques au maximum ;
   - `ReduceLROnPlateau` : learning rate divisé par 2 après 3 époques sans amélioration de la macro-F1 de validation ;
   - `EarlyStopping` : arrêt après 8 époques sans amélioration de la macro-F1 de validation, puis restauration des meilleurs poids.
5. **Analyse des résultats** : matrice de confusion, rapport de classification, exemples de mauvaises prédictions.
6. **3 expériences** sur l'architecture du CNN.
7. **Data augmentation** pour tenter d'améliorer l'apprentissage du CNN.
8. **YOLO + CNN** : détection de plusieurs visages, puis classification de chacun.
9. **Extension à la vidéo**.

### Pré-requis

- **Tokens Kaggle** : dans Colab, créer les secrets `KAGGLE_USERNAME` et `KAGGLE_KEY` (onglet « Secrets » de la barre latérale, avec « Accès depuis le notebook » activé). Le token se génère sur Kaggle via « Your API tokens » → « Generate New Token ».
- **Une vidéo `.mp4`** (facultatif, uniquement pour la partie 9). Elle n'est pas fournie dans le dépôt.
- **(Optionnel) Modèles pré-entraînés** : le dossier [models/](models/) contient un modèle entraîné pour chaque partie. Il suffit d'exécuter les cellules « (Optionnel) charger le modèle pré-entraîné » au lieu des cellules d'entraînement.

| Fichier                   | Modèle                              |
| :------------------------ | :---------------------------------- |
| `mlp_baseline.keras`      | Modèle de référence MLP (partie 2)  |
| `cnn_model_base.keras`    | CNN de base (parties 3 et 4)        |
| `cnn_exp_1.keras`         | Expérience 1 (partie 6)             |
| `cnn_exp_2.keras`         | Expérience 2 (partie 6)             |
| `cnn_exp_3.keras`         | Expérience 3 (partie 6)             |
| `cnn_data_augm.keras`     | CNN avec data augmentation (partie 7) |

## Architecture

### Modèle de référence (partie 2)

Un MLP volontairement simple, qui n'exploite pas la structure spatiale de l'image :

| Couche  | Détail                | Sortie         |
| :------ | :-------------------- | :------------- |
| Input   | image RGB             | (100, 100, 3)  |
| Flatten | —                     | (30 000)       |
| Dense   | 128 neurones, ReLU    | (128)          |
| Dense   | 7 neurones, Softmax   | (7)            |

### CNN (partie 3)

Trois blocs `Conv2D → BatchNormalization → ReLU → MaxPooling2D`, dont le nombre de filtres double à chaque bloc (32 → 64 → 128), suivis d'une couche de classification :

| Bloc       | Couches                                                       | Sortie          |
| :--------- | :------------------------------------------------------------ | :-------------- |
| Entrée     | —                                                             | (100, 100, 3)   |
| Bloc 1     | Conv2D 32 filtres 3×3 (`same`), BatchNorm, ReLU, MaxPool 2×2  | (50, 50, 32)    |
| Bloc 2     | Conv2D 64 filtres 3×3 (`same`), BatchNorm, ReLU, MaxPool 2×2  | (25, 25, 64)    |
| Bloc 3     | Conv2D 128 filtres 3×3 (`same`), BatchNorm, ReLU, MaxPool 2×2 | (12, 12, 128)   |
| Sortie     | Flatten, Dense 7 + Softmax                                    | (7)             |

Les convolutions n'ont pas de biais (`use_bias=False`), car celui-ci est redondant avec le paramètre beta de la BatchNormalization.

## Expériences

Chaque expérience reprend le CNN de base et la même boucle d'entraînement, en ne modifiant qu'un seul élément :

1. **Type de pooling** : `MaxPooling2D` remplacé par `AveragePooling2D`.
2. **Taille de la fenêtre de pooling** : 2×2 remplacée par 4×4, ce qui réduit beaucoup plus vite la résolution spatiale.
3. **Dropout** : ajout d'une couche Dense intermédiaire de 256 neurones, suivie d'une BatchNormalization et d'un `Dropout(0.5)` avant la sortie.

Les courbes d'accuracy, de loss et de macro-F1 de chaque modèle sont dans [images/comparaisons/](images/comparaisons/).

## Résultats

| Modèle                     | Accuracy (test) | Macro-F1 (test) |
| :------------------------- | --------------: | --------------: |
| CNN de base                |           0.782 |           0.681 |
| CNN avec data augmentation |           0.790 |           0.704 |

- **Le modèle surapprend** : l'accuracy d'entraînement atteint presque 100 % alors que celle de validation plafonne autour de 77 %, et la loss de validation remonte au fil des époques.
- **Les 3 expériences ne changent presque rien** : les courbes diffèrent légèrement, mais les métriques finales sont quasi identiques. Le Dropout ralentit un peu le surapprentissage, sans améliorer le score.
- **La data augmentation aide un peu** : un retournement horizontal et un contraste aléatoires réduisent le surapprentissage (loss de validation plus basse) et améliorent le recall de certaines classes, mais le gain reste très faible.
- Les classes les mieux reconnues sont la Joie, la Surprise et Neutre. La Peur et le Dégoût sont les plus difficiles : ce sont des classes minoritaires, et leurs expressions ressemblent à d'autres (la Peur est souvent confondue avec la Surprise).

## Extension YOLO et vidéo

### Image (partie 8)

Un modèle YOLOv8 pré-entraîné à la détection de visages ([arnabdhar/YOLOv8-Face-Detection](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection)) est téléchargé depuis Hugging Face. Le pipeline est le suivant :

1. YOLO détecte les visages et renvoie leurs bounding boxes ;
2. chaque visage est découpé, redimensionné en 100×100 RGB et normalisé dans [0, 1], comme les images d'entraînement ;
3. le CNN prédit l'émotion de chaque visage ;
4. l'image est affichée avec une boîte et l'émotion prédite (et sa probabilité) pour chaque visage.

La fonction `detect_and_classify(image_path)` regroupe toute cette pipeline. Une image d'exemple est fournie : [images/sample_yolo_image.jpg](images/sample_yolo_image.jpg).

### Vidéo (partie 9)

Le même pipeline est appliqué image par image : `ffmpeg` découpe la vidéo en frames (5 fps), chaque frame est annotée, puis `ffmpeg` reconstruit la vidéo finale (`/content/video_resultat.mp4`).

**La vidéo n'est pas fournie** : il faut importer soi-même un fichier `.mp4` dans Colab et adapter la variable `video_path` (par défaut `/content/images/video.mp4`).
