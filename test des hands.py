import cv2
import mediapipe as mp
import cvzone
from cvzone.HandTrackingModule import HandDetector

originex, originey = 203, 128

# Créer un détecteur de main
detector = HandDetector(staticMode=False, maxHands=2, detectionCon=0.85, minTrackCon=0.5)

# Ouvrir la webcam
Capture = cv2.VideoCapture(0)

# Boucle infinie pour afficher la vidéo en direct
while True:
    stat, image = Capture.read()
    image = cv2.flip(image, 1)  # Effet miroir
    h, w, _ = image.shape

    # Convertir l'image en format RGB (utilisé par MediaPipe)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Détecter les mains
    hands, img = detector.findHands(image)

    # Dessiner un rectangle transparent
    overlay = image.copy()
    _, rect = cvzone.putTextRect(
        overlay,
        " ",  # Texte vide ici
        (originex, originey),
        10,
        3,
        (240, 30, 30),
        (46, 81, 240),
        cv2.FONT_HERSHEY_PLAIN,
        30,
        2,
        (0, 255, 0))
    alpha = 0.8
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Si une main est détectée
    if hands:
        for hand in hands:
            lmList = hand["lmList"]  # Liste des points clés
            x1, y1 = lmList[8][0], lmList[8][1]   # Index
            x2, y2 = lmList[12][0], lmList[12][1] # Majeur

            # Calculer et dessiner la distance entre les deux doigts
            length, _, image = detector.findDistance((x1, y1), (x2, y2), image)

    # Afficher l'image dans une fenêtre
    cv2.imshow("image", image)

    # Fermer la fenêtre si on appuie sur la touche 'a'
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

# Fermer la webcam et la fenêtre
Capture.release()
cv2.destroyAllWindows()
