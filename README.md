# NM i AI 2026

Dette er et av de første AI-prosjektene mine i NM i AI 2026.

Her har jeg jobbet med et bildedatasett med produkter og provd aa trene en enkel YOLO-modell som kan kjenne igjen varer i bilder. Maalet mitt har vaert aa laere hvordan et AI-prosjekt faktisk ser ut i praksis: se paa data, forstaa annotasjoner, lage et treningssett og kjoere min forste modell.

## Hva prosjektet inneholder

- `images/` inneholder bildene
- `annotations.json` inneholder annotasjonene i COCO-format
- `scripts/analyze_dataset.py` viser en enkel oppsummering av datasettet
- `scripts/export_yolo_subset.py` lager et mindre YOLO-datasett
- `scripts/pretty_print_results.py` skriver ut treningsresultater fra `results.csv` paa en mer lesbar maate

## Hva jeg gjorde for aa komme i gang

Det forste jeg gjorde var aa se paa hva datasettet faktisk inneholdt:

```bash
python3 scripts/analyze_dataset.py
```

Da fikk jeg blant annet vite:

- hvor mange bilder datasettet hadde
- hvor mange annotasjoner det hadde
- hvor mange klasser som fantes

Jeg laerte raskt at det ikke var lurt aa starte med alle klassene samtidig, saa jeg lagde et mindre datasett med de vanligste produktene.

## Lage et mindre treningssett

```bash
python3 scripts/export_yolo_subset.py --top-k 10
```

Dette lager et YOLO-datasett i:

- `exports/yolo_top10/images/train`
- `exports/yolo_top10/images/val`
- `exports/yolo_top10/labels/train`
- `exports/yolo_top10/labels/val`
- `exports/yolo_top10/data.yaml`

Standardoppsettet:

- velger de 10 vanligste klassene
- hopper over `unknown_product`
- splitter data i `train` og `val`

## Trene en enkel modell

Jeg brukte Ultralytics YOLO for aa trene en liten modell foerst.

Hvis YOLO ikke er installert:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics
```

For aa starte en enkel trening:

```bash
yolo detect train data=exports/yolo_top10/data.yaml model=yolov8n.pt epochs=10 imgsz=640
```

Senere kan jeg kjoere flere epochs, for eksempel:

```bash
yolo detect train data=exports/yolo_top10/data.yaml model=yolov8n.pt epochs=30 imgsz=640
```

## Hva jeg har laert sa langt

- COCO-annotasjoner er delt opp i `images`, `annotations` og `categories`
- YOLO trenger data i et annet format enn det opprinnelige datasettet
- det er smartere aa starte lite enn aa prove alt paa en gang
- treningsresultater maa tolkes, ikke bare kjares

## Nyttige kommandoer

Se datasettet:

```bash
python3 scripts/analyze_dataset.py
```

Lag et stoerre lite datasett:

```bash
python3 scripts/export_yolo_subset.py --top-k 20
```

Ta med `unknown_product`:

```bash
python3 scripts/export_yolo_subset.py --top-k 10 --include-unknown
```

Skriv ut treningsresultater penere:

```bash
python3 scripts/pretty_print_results.py
```

## Videre arbeid

Neste steg for meg er aa:

- sammenligne korte treningsrunder med ulike innstillinger
- forstaa resultatene bedre, som `precision`, `recall` og `mAP`
- prove aa forbedre modellen litt etter litt

Dette repoet er derfor ikke bare et datasett eller et kodeprosjekt, men ogsaa en del av laeringsprosessen min i AI.
