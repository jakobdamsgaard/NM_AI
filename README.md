# NM i AI starter

Dette repoet inneholder et COCO-merket bildedatasett:

- `images/` har bildene
- `annotations.json` har bokser og produktnavn

Hvis du er ny, ikke start med alle 356 klassene samtidig. Start med et mindre YOLO-datasett basert pa de vanligste produktene.

## 1. Se hva du har

Kjor:

```bash
python3 scripts/analyze_dataset.py
```

Det viser hvor mange bilder, annotasjoner og klasser du har, samt de vanligste klassene.

## 2. Lag et lite treningssett

Kjor:

```bash
python3 scripts/export_yolo_subset.py --top-k 10
```

Dette lager:

- `exports/yolo_top10/images/train`
- `exports/yolo_top10/images/val`
- `exports/yolo_top10/labels/train`
- `exports/yolo_top10/labels/val`
- `exports/yolo_top10/data.yaml`

Standardvalg:

- tar de 10 vanligste klassene
- hopper over `unknown_product`
- splitter bilder i train/val med fast seed

## 3. Tren en YOLO-modell

Etter at du har installert YOLO lokalt eller i Colab, kan du trene med:

```bash
yolo detect train data=exports/yolo_top10/data.yaml model=yolov8n.pt epochs=30 imgsz=640
```

Hvis `yolo` ikke finnes ennå, installer Ultralytics i et virtuelt miljo:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics
```

## 4. Neste steg

Nar flyten virker:

- ok `--top-k 20`
- tren flere epoker
- vurder om `unknown_product` skal inkluderes
- se pa feilklassifiseringer for a rydde opp i datasettet

## Nyttige kommandoer

Lag et sturre datasett:

```bash
python3 scripts/export_yolo_subset.py --top-k 20
```

Ta med `unknown_product`:

```bash
python3 scripts/export_yolo_subset.py --top-k 10 --include-unknown
```
