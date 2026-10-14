# YOLOv4 (Pascal VOC 2012) – Industrial PyTorch Implementation

Eine modulare, produktionsreife PyTorch-Implementierung von **YOLOv4: Optimal Speed and Accuracy of Object Detection** *(Bochkovskiy, Wang, Liao, 2020)* für den Pascal VOC 2012 Datensatz.

Dieses Projekt folgt modernen Software-Engineering- und MLOps-Standards der Industrie:
- **Entkoppelte Konfigurationen:** Strukturierte Parameter & Hyperparameter via YAML (`configs/`).
- **Modulare Modellarchitektur:** Klare Trennung von Backbone, Neck (SPP + PANet) und Detection Head (`models/`).
- **Modernes Verlust- & Datenparadigma (Bag of Freebies):** CIoU Loss, Mosaic Data Augmentation, Label Smoothing.
- **Experiment-Tracking & MLOps:** Nahtlose Integration von **Weights & Biases (`wandb`)** für Metrik- und Bild-Logging.
- **Production Deployment:** Export nach **ONNX** und **TorchScript** sowie Post-Training-Quantisierung (**FP16 & INT8**).

---

## 📁 Projektstruktur

```text
03_YOLOv4/
├── configs/                  # Konfigurationsdateien (Single Source of Truth)
│   ├── voc.yaml              # Datensatz-Pfade, Klassenanzahl & Klassennamen
│   └── hyp.yaml              # Hyperparameter (LR, Augmentierung, Loss-Gewichte, Anchors)
│
├── data/                     # Datenpipeline & Vorverarbeitung
│   ├── __init__.py
│   ├── dataset.py            # VOCDataset (XML-Parsing, Ground-Truth Bounding Boxes)
│   ├── augmentations.py      # Albumentations-Pipeline & Mosaic 4-Image-Augmentation
│   └── dataloader.py         # PyTorch DataLoader & custom collate_fn
│
├── models/                   # Modulare Netzwerkarchitektur
│   ├── __init__.py
│   ├── common.py             # Basis-Layer: ConvBnMish, ResidualBlock, CSPBlock, SPPBlock
│   ├── backbone.py           # CSPDarknet-53 Feature Extractor (3 Ausgangs-Skalen)
│   ├── neck.py               # SPP + PANet (Top-Down FPN + Bottom-Up Path Aggregation)
│   ├── head.py               # YOLOv4 Multi-Scale Detection Head
│   └── yolo.py               # Zusammengesetztes Gesamtsystem (YOLOv4 nn.Module)
│
├── utils/                    # Mathematische Helfer & Utility-Funktionen
│   ├── __init__.py
│   ├── boxes.py              # Box-Konvertierungen (xyxy <-> xywh), IoU, Complete-IoU (CIoU), NMS
│   ├── metrics.py            # Average Precision (AP), mAP@0.5, mAP@0.5:0.95
│   ├── torch_utils.py        # Model EMA (Exponential Moving Average), Profiling & FLOPs
│   └── plots.py              # Visualisierung von Vorhersagen und W&B Artifacts
│
├── tests/                    # Automatisierte Unit-Tests & Shape-Checks
│   ├── test_model.py         # Forward-Pass & Tensor-Shape-Verifikation
│   └── test_loss.py          # Stabilitätsprüfung des CIoU-Loss
│
├── loss.py                   # Multi-Scale Loss: CIoU Coordinate Loss + BCE Objectness & Class
├── train.py                  # Training Engine mit Mixed Precision (AMP), EMA, W&B und Checkpointing
├── evaluate.py               # Standalone Evaluation auf dem Validierungsset
├── predict.py                # Inferenz-Pipeline (Einzelbild, Batch, Video, FPS-Messung)
├── export.py                 # Export-Pipeline: PyTorch -> ONNX / TorchScript & INT8-Quantisierung
└── README.md                 # Projektdokumentation & Entwicklungs-Roadmap
```

---

## 🗺️ Der 3-Phasen-Entwicklungsplan (Roadmap)

Um das Projekt methodisch wie in einem professionellen Entwicklungsteam aufzubauen, gliedert sich die Umsetzung in drei aufeinander aufbauende Phasen:

```
[ Phase 1: Core Foundation ] ──▶ [ Phase 2: Training & Logging ] ──▶ [ Phase 3: Deployment & Export ]
  • YAML Configs                   • Training Engine (AMP, EMA)        • Inferenz-Pipeline (predict.py)
  • Box-Math & CIoU                • Weights & Biases Logging          • ONNX & TorchScript Export
  • Dataset & Mosaic               • mAP-Evaluierung                   • FP16 & INT8 Quantisierung
  • CSPDarknet53 + PANet           • Checkpoint & Resume               • FPS- & Latenz-Benchmarks
  • Overfit 1-Batch Test
```

### Phase 1: Core Foundation & Verifikation
*Ziel: Die mathematischen und architektonischen Grundlagen schaffen und isoliert verifizieren.*
1. **Konfiguration:**
   - `configs/voc.yaml`: Definition der Pfade zu Pascal VOC 2012, Anzahl der Klassen (`nc: 20`) und Klassennamen.
   - `configs/hyp.yaml`: Hyperparameter (Learning Rate, Optimizer, Loss-Gewichte, Anchors für Skalen 8, 16, 32).
2. **Mathematik & Geometrie (`utils/boxes.py`):**
   - Vektorisierte Funktionen für Box-Umrechnungen: `xywh` $\leftrightarrow$ `xyxy`.
   - Implementierung von **Intersection over Union (IoU)** und **Complete IoU (CIoU)**.
   - Target-Encoding: Zuordnung von Ground-Truth-Boxen zu den 3 Feature-Grids und Anchors.
3. **Daten-Pipeline (`data/`):**
   - `dataset.py`: Parsen der VOC-XML-Annotationen.
   - `augmentations.py`: Implementierung der **Mosaic-Augmentation** (Zusammenfügen von 4 zufälligen Bildern) und Albumentations (Farbe, Flipping).
   - `dataloader.py`: Benutzerdefinierte `collate_fn` zur Handhabung variabler Box-Längen pro Bild.
4. **Modell-Architektur (`models/`):**
   - `common.py`: Implementierung von `ConvBnMish` (Mish-Aktivierung: $x \cdot \tanh(\ln(1 + e^x))$) und `CSPBlock`.
   - `backbone.py`: Aufbau von **CSPDarknet-53** mit 3 hierarchischen Feature-Ausgängen ($52\times52, 26\times26, 13\times13$ bei $416\times416$ Input).
   - `neck.py`: **SPP** (Spatial Pyramid Pooling: MaxPool mit $1\times1, 5\times5, 9\times9, 13\times13$) + **PANet** (Top-down FPN kombiniert mit Bottom-up Path Aggregation).
   - `head.py` & `yolo.py`: Zusammenfügen zu den 3 Detection Heads.
5. **Verlustfunktion (`loss.py`):**
   - Multi-Scale Loss:
     - Koordinaten: **CIoU Loss** auf positiven Ankern.
     - Objekt-Konfidenz: **Binary Cross-Entropy (BCE)** mit Ignore-Schwellenwert (IoU $> 0.5$).
     - Klassifikation: **BCE** pro Klasse (Multi-Label) mit optionalem Label-Smoothing.
6. **Qualitätssicherung (Meilenstein 1 Test):**
   - Unit-Tests (`tests/test_model.py`) für Shape-Konsistenz.
   - **Overfitting-Test:** Training auf genau *einem* Mini-Batch – der Loss muss zuverlässig gegen 0 fallen.

---

### Phase 2: Training Pipeline, MLOps & Evaluation
*Ziel: Ein stabiles, reproduzierbares Training auf dem gesamten Datensatz mit lückenlosem Monitoring.*
1. **Training Engine (`train.py`):**
   - **Automatic Mixed Precision (AMP):** `torch.amp.autocast` und `GradScaler` für maximale GPU-Performance.
   - **Model EMA (`utils/torch_utils.py`):** Führen eines exponentiell geglätteten Gewichts-Zustands für stabilere Validierungs-Ergebnisse.
   - **Lernraten-Steuerung:** Linearer Warmup gefolgt von Cosine-Annealing.
   - **Robustes Checkpointing:** Speichern von `best.pth` und `last.pth` inklusive Optimizer-, Scheduler- und Epochen-Status für nahtloses `--resume`.
2. **MLOps & Experiment-Tracking:**
   - **Weights & Biases (`wandb`):**
     - Live-Logging von Train/Val Loss, Learning Rate, Box-Loss, Class-Loss, Objectness-Loss.
     - Automatisches Hochladen von Visualisierungs-Batches mit Ground-Truth vs. Prediction Bounding Boxes.
3. **Evaluation (`evaluate.py`):**
   - Nicht-maximale Unterdrückung (**NMS**) zur Filterung überlappender Boxen.
   - Berechnung von $\text{mAP@0.5}$ (Pascal VOC Metrik) und $\text{mAP@0.5:0.95}$ (COCO-Standard).
   - Per-Class Average Precision (AP) Übersicht.

---

### Phase 3: Production Deployment & Model Optimization
*Ziel: Maximale Performance und Unabhängigkeit von der Python-Entwicklungsumgebung.*
1. **Inferenz-Pipeline (`predict.py`):**
   - Laden trainierter Gewichte.
   - Unterstützung für Einzelbilder, Verzeichnisse, Videos und Live-Webcam.
   - End-to-End Latenz- und FPS-Messung unter Berücksichtigung von GPU-Synchronisation (`torch.cuda.synchronize()`).
2. **Export-Pipeline (`export.py`):**
   - **ONNX Export:** Umwandlung des PyTorch-Graphs in das herstellerneutrale ONNX-Format.
   - **TorchScript Export:** Tracing/Scripting für den Einsatz in C++ Umgebungen (`libtorch`).
3. **Modell-Quantisierung & Beschleunigung:**
   - **FP16 Half-Precision Inferenz:** Halbierung des Speicherbedarfs bei nahezu identischer Genauigkeit.
   - **INT8 Post-Training Quantization (PTQ):** 8-Bit-Ganzzahl-Quantisierung zur Beschleunigung auf CPUs und Edge-Geräten.
   - Erstellung einer Vergleichstabelle: Modellgröße (MB), Inferenzzeit (ms) und mAP für FP32 vs. FP16 vs. INT8.

---

## 📐 Mathematischer Kern: Der Complete-IoU (CIoU) Loss

Im Gegensatz zu YOLOv3 (MSE auf Koordinaten) nutzt YOLOv4 den **CIoU Loss**, welcher drei geometrische Kernfaktoren direkt optimiert:
1. **Überlappungsfläche (IoU):** $\text{IoU} = \frac{|B \cap B^{gt}|}{|B \cup B^{gt}|}$
2. **Zentrumsabstand:** $\frac{\rho^2(b, b^{gt})}{c^2}$ (wobei $\rho$ die euklidische Distanz der Mittelpunkte und $c$ die Diagonale des kleinsten umschließenden Rechtecks ist)
3. **Seitenverhältnis-Konsistenz:** $\alpha v$, wobei:
   $$v = \frac{4}{\pi^2} \left( \arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h} \right)^2$$
   $$\alpha = \frac{v}{(1 - \text{IoU}) + v}$$

Die Verlustfunktion lautet:
$$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

---

## 🛠️ Abhängigkeiten & Installation

```bash
pip install torch torchvision
pip install albumentations pyyaml wandb tqdm matplotlib pillow
pip install onnx onnxruntime pytest
```

---

## 💡 Status & Nächste Schritte

- [x] Verzeichnis- & Dateistruktur angelegt
- [x] Entwicklungs-Roadmap & Architektur spezifiziert (`README.md`)
- [ ] **Schritt 1:** Konfigurationsdateien ausfüllen (`configs/voc.yaml` & `configs/hyp.yaml`)
- [ ] **Schritt 2:** Mathematische Helfer in `utils/boxes.py` implementieren
- [ ] **Schritt 3:** Datenpipeline & Mosaic-Augmentation in `data/` erstellen
- [ ] **Schritt 4:** Modellkomponenten in `models/` bauen
- [ ] **Schritt 5:** CIoU Loss in `loss.py` schreiben und 1-Batch-Overfit-Test durchführen
