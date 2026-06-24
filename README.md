# Aplicativo de Reconhecimento Facial com Webcam

Este projeto implementa um app em Python para:

- Capturar imagens de rostos pela webcam
- Treinar um modelo de reconhecimento facial (LBPH)
- Reconhecer pessoas em tempo real

## Requisitos

- Python 3.10+
- Webcam conectada

## Instalação

```bash
pip install -r requirements.txt
```

## Como usar

### 1) Capturar imagens de uma pessoa

```bash
python app.py capture --name Joao --samples 60
```

- Pressione `q` para encerrar antes de completar as amostras.

### 2) Treinar o modelo

```bash
python app.py train
```

### 3) Reconhecer em tempo real

```bash
python app.py recognize --threshold 70
```

- Pressione `q` para sair.

## Dicas

- Capture imagens de cada pessoa em diferentes ângulos e iluminação.
- Se o reconhecimento estiver ruim, aumente o número de amostras e treine novamente.
- O valor de `threshold` controla rigor da identificação (menor = mais restrito).

## Estrutura

- `app.py`: lógica principal
- `data/dataset`: imagens capturadas
- `data/trainer/model.yml`: modelo treinado
- `data/trainer/labels.json`: mapeamento de IDs para nomes
