import argparse
import json
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "data" / "dataset"
TRAINER_DIR = BASE_DIR / "data" / "trainer"
MODEL_PATH = TRAINER_DIR / "model.yml"
LABELS_PATH = TRAINER_DIR / "labels.json"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def ensure_dirs() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    TRAINER_DIR.mkdir(parents=True, exist_ok=True)


def get_face_detector() -> cv2.CascadeClassifier:
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError("Nao foi possivel carregar o classificador Haar Cascade.")
    return detector


def get_recognizer():
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise RuntimeError(
            "OpenCV contrib nao encontrado. Instale com: pip install opencv-contrib-python"
        )
    return cv2.face.LBPHFaceRecognizer_create()


def capture_faces(name: str, samples: int = 50) -> None:
    ensure_dirs()
    detector = get_face_detector()

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Nao foi possivel abrir a webcam.")

    print(f"Capturando rosto de '{name}'...")
    print("Pressione 'q' para sair.")

    count = 0
    while count < samples:
        ok, frame = cam.read()
        if not ok:
            print("Falha ao ler frame da webcam.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y : y + h, x : x + w]
            face_img = cv2.resize(face_img, (200, 200))

            file_path = DATASET_DIR / f"{name}.{count}.jpg"
            cv2.imwrite(str(file_path), face_img)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 200, 50), 2)
            cv2.putText(
                frame,
                f"{count}/{samples}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (50, 200, 50),
                2,
            )

            if count >= samples:
                break

        cv2.imshow("Captura - Reconhecimento Facial", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

    print(f"Captura finalizada. Amostras salvas: {count}")


def parse_dataset():
    faces = []
    labels = []
    name_to_id = {}

    image_paths = sorted(DATASET_DIR.glob("*.jpg"))
    if not image_paths:
        raise RuntimeError("Nenhuma imagem encontrada em data/dataset.")

    next_id = 0
    for img_path in image_paths:
        parts = img_path.stem.split(".")
        if len(parts) < 2:
            continue

        name = parts[0]
        if name not in name_to_id:
            name_to_id[name] = next_id
            next_id += 1

        label_id = name_to_id[name]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        faces.append(img)
        labels.append(label_id)

    if not faces:
        raise RuntimeError("Nao foi possivel carregar imagens validas para treino.")

    return faces, np.array(labels), name_to_id


def train_model() -> None:
    ensure_dirs()
    recognizer = get_recognizer()
    faces, labels, name_to_id = parse_dataset()

    recognizer.train(faces, labels)
    recognizer.write(str(MODEL_PATH))

    id_to_name = {str(v): k for k, v in name_to_id.items()}
    LABELS_PATH.write_text(json.dumps(id_to_name, ensure_ascii=True, indent=2), encoding="utf-8")

    print("Treinamento concluido.")
    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"Labels salvos em: {LABELS_PATH}")


def recognize_faces(threshold: float = 70.0) -> None:
    ensure_dirs()
    detector = get_face_detector()
    recognizer = get_recognizer()

    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise RuntimeError("Modelo nao encontrado. Rode: python app.py train")

    recognizer.read(str(MODEL_PATH))
    id_to_name = json.loads(LABELS_PATH.read_text(encoding="utf-8"))

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Nao foi possivel abrir a webcam.")

    print("Reconhecimento iniciado. Pressione 'q' para sair.")

    while True:
        ok, frame = cam.read()
        if not ok:
            print("Falha ao ler frame da webcam.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face_img = gray[y : y + h, x : x + w]
            face_img = cv2.resize(face_img, (200, 200))

            pred_id, confidence = recognizer.predict(face_img)

            if confidence <= threshold:
                name = id_to_name.get(str(pred_id), "Desconhecido")
                label = f"{name} ({confidence:.1f})"
                color = (60, 220, 60)
            else:
                label = f"Desconhecido ({confidence:.1f})"
                color = (50, 50, 230)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        cv2.imshow("Reconhecimento Facial", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="App de reconhecimento facial com webcam")
    sub = parser.add_subparsers(dest="command")

    p_capture = sub.add_parser("capture", help="Captura amostras de uma pessoa")
    p_capture.add_argument("--name", required=True, help="Nome da pessoa")
    p_capture.add_argument("--samples", type=int, default=50, help="Quantidade de amostras")

    sub.add_parser("train", help="Treina o modelo com as imagens capturadas")

    p_recognize = sub.add_parser("recognize", help="Reconhece pessoas em tempo real")
    p_recognize.add_argument(
        "--threshold",
        type=float,
        default=70.0,
        help="Limite de confianca (menor = mais estrito)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "capture":
            capture_faces(name=args.name, samples=max(1, args.samples))
        elif args.command == "train":
            train_model()
        elif args.command == "recognize":
            recognize_faces(threshold=args.threshold)
        else:
            parser.print_help()
    except Exception as exc:
        print(f"Erro: {exc}")


if __name__ == "__main__":
    main()
