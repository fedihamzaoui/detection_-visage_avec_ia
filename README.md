Ce projet utilise OpenCV et des bibliothèques optionnelles comme MediaPipe, face_recognition et TensorFlow pour créer une application de caméra en temps réel capable de :

Détecter et reconnaître les visages connus.

Estimer l'émotion (heuristique et modèle avancé).

Estimer l'âge et le genre de la personne.

Détecter les sourires et calculer une intensité approximative.

Estimer la pose de la tête (orientation) via des repères approximatifs.

Mesurer la fréquence cardiaque (rPPG) à partir du front si SciPy est installé.

Détecter les mains et la posture avec MediaPipe (si disponible).

Fonctionnement

Ouvre la caméra par défaut et affiche le flux en temps réel.

Les informations détectées sont affichées directement sur l’image.

Appuyer sur s pour sauvegarder une capture et q pour quitter.

Bibliothèques utilisées

OpenCV pour la capture vidéo et la détection de visages/yeux/sourires.

MediaPipe pour le suivi des mains et de la posture (optionnel).

face_recognition pour la reconnaissance faciale (optionnel).

TensorFlow pour les modèles avancés d’émotion, âge et genre (optionnel).

SciPy pour le traitement rPPG et l’estimation de la fréquence cardiaque (optionnel).
