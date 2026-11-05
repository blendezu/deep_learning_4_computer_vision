# Swin Transformer V2: Lokales Training & Inferenz (Food-101)

Dieses Verzeichnis enthält den vollständig modularisierten Python-Code zur lokalen Ausführung, zum Training und zur visuellen Erklärbarkeit (GradCAM) des **Swin Transformer V2 (Swin-T)**, basierend auf dem Jupyter Notebook [`swinV2_from_scratch.ipynb`](../swinV2_from_scratch.ipynb).

---

## 🌟 Kernkonzepte von Swin Transformer V2

Im Vergleich zu Swin V1 führt Swin V2 drei wesentliche Architektur-Verbesserungen ein:
1. **Res-Post-Norm (Residual Post-Normalization)**: Normalisierungsschichten werden hinter die Attention- und MLP-Blöcke platziert, was die numerische Stabilität bei tiefen Netzen stark erhöht.
2. **Cosine Attention mit skalierbarem Logit-Faktor**: Statt standardmäßigem Skalarprodukt verwendet W-MSA eine Kosinus-Ähnlichkeit mit lernbarem `logit_scale`, wodurch extreme Aufmerksamkeitswerte verhindert werden.
3. **Log-Spaced Continuous Relative Position Bias (CPB-MLP)**: Relative Positionsbiases werden über ein kleines 2-Schicht-MLP aus logarithmierten Koordinaten generiert, was einen flexiblen Transfer über verschiedene Fenster- und Bildgrößen ermöglicht.

---

## 📁 Projektstruktur

```text
local/
├── config.py           # Zentrale Konfiguration (Hyperparameter, Pfade, Normalisierung, Seed)
├── dataset.py          # Food-101 DataLoaders, Datenaugmentation, CutMix & MixUp
├── model.py            # Vollständige Swin Transformer V2 Architektur von Grund auf (Mlp, WindowAttention, SwinBlock, etc.)
├── utils.py            # Seeding, EarlyStopping und Lernkurven-Visualisierung
├── train.py            # Haupt-Trainingsskript mit CLI-Argumenten, AMP (Mixed Precision) & OneCycleLR
├── gradcam.py          # GradCAM Erklärbarkeitsanalyse mit Feature-Reshaping (B, L, C -> B, C, H, W)
├── predict.py          # Inferenzskript für Einzelbilder mit Top-K Vorhersagen und GradCAM-Option
├── requirements.txt    # Benötigte Python-Bibliotheken
├── checkpoints/        # Speicherort für Modell-Gewichte (z. B. best_checkpoint.pth)
└── figures/            # Exportierte Plots (training_history.png, gradcam_sample.png)
```

---

## ⚙️ Voraussetzungen & Installation

Erstelle eine Python-Umgebung (empfohlen: Python 3.10 oder 3.11) und installiere die Abhängigkeiten:

```bash
pip install -r requirements.txt
```

Für NVIDIA-Grafikkarten wird PyTorch mit CUDA-Unterstützung empfohlen:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 🚀 Training starten

Das Training kann direkt über `train.py` ausgeführt werden:

```bash
# Standard-Training mit Swin-T starten (lädt Food-101 automatisch herunter)
python train.py
```

### Wichtige Parameter & Optionen:

| Argument | Standardwert | Beschreibung |
| :--- | :--- | :--- |
| `--epochs` | `2` | Anzahl der Epochen |
| `--batch-size` | `128` | Batch-Größe für Training und Validierung |
| `--lr` | `1e-3` | Maximale Lernrate für den `OneCycleLR`-Scheduler |
| `--model-name` | `Swin-T` | Architektur-Variante (`Swin-T`, `Swin-S`, `Swin-B`, `Swin-L`) |
| `--img-size` | `224` | Eingabebildgröße in Pixeln |
| `--weight-decay`| `0.05` | Weight Decay für AdamW |
| `--patience` | `10` | Geduld in Epochen für Early Stopping |
| `--delta` | `0.01` | Mindestverbesserung der Validation Loss |
| `--data-path` | `./Data` | Lokaler Speicherpfad für den Food-101 Datensatz |
| `--num-workers` | `4` | Anzahl paralleler Worker für Daten-Prefetching |
| `--no-amp` | `False` | Deaktiviert Automatic Mixed Precision (AMP) |
| `--checkpoint-path` | `checkpoints/best_checkpoint.pth` | Zielpfad des besten Modells |
| `--save-plot` | `figures/training_history.png` | Zielpfad für den Trainingsverlaufs-Plot |

### Beispiele:

```bash
# Schneller Probelauf mit kleinerer Batch-Größe
python train.py --epochs 3 --batch-size 32

# Training auf separatem Datensatzpfad ohne AMP
python train.py --data-path "C:/Datasets/Food101" --no-amp
```

---

## 🔬 GradCAM Visualisierung

Mit `gradcam.py` wird die visuelle Entscheidungsgrundlage des Transformators analysiert. Da Swin Transformer mit sequentiellen Token-Tensoren `(B, L, C)` arbeitet, re-projiziert das Modul die Aktivierungen und Gradienten in 2D Feature-Maps `(B, C, H, W)`:

```bash
# GradCAM für ein Testbeispiel aus Food-101 erzeugen
python gradcam.py --checkpoint checkpoints/best_checkpoint.pth

# GradCAM für ein eigenes Bild erzeugen
python gradcam.py --image "mein_gericht.jpg" --checkpoint checkpoints/best_checkpoint.pth
```
Das Ergebnis wird standardmäßig unter `figures/gradcam_sample.png` abgespeichert.

---

## 🔍 Inferenz & Vorhersagen

Um ein beliebiges Bild zu klassifizieren und die Top-5 wahrscheinlichsten Klassen zu erhalten:

```bash
python predict.py --image "pfad/zu/pizza.jpg" --checkpoint checkpoints/best_checkpoint.pth
```

Mit optionaler GradCAM-Erklärung:
```bash
python predict.py --image "pfad/zu/pizza.jpg" --checkpoint checkpoints/best_checkpoint.pth --gradcam
```
