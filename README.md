# Teemo Recommendation System

This project implements a Two-Tower recommendation model for matching competitions with users based on their personas.

## Project Structure

- `model_training.ipynb`: Notebook for training the recommendation model using synthetic user data.
- `inference.py`: Standalone module for generating top-N recommendations for a given user profile.
- `main.py`: FastAPI service providing a REST API for the recommendation system.
- `hasil_feature.csv`: Real competition data.
- `synthetic_users.csv`: Synthetic user personas for training.
- `teemo_model.keras`: Trained TensorFlow model (produced after training).
- `scaler.pkl`: Saved MinMaxScaler (produced after training).

## Requirements

This project uses `uv` for lightning-fast Python package management.

### Prerequisites

- Python 3.10 or newer
- [uv](https://github.com/astral-sh/uv) installed on your system

## Getting Started

### 1. Setup Environment

Initialize the virtual environment and install dependencies:

```bash
uv venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows
uv pip install -r requirements.txt numpy pydantic ipykernel
```

Alternatively, you can just run:

```bash
uv sync
```

### 2. Train the Model

Open `model_training.ipynb` in your favorite Jupyter environment (e.g., VS Code) and run all cells. This will:
- Load the competition and synthetic user data.
- Generate ~15,000 training pairs based on persona matching logic.
- Train a Two-Tower neural network.
- Save `teemo_model.keras` and `scaler.pkl` in the root directory.

### 3. Run Inference (Local Test)

You can test the recommendation logic directly:

```bash
python inference.py
```

This will run a test for the "Tech Mahasiswa Online" persona and print the top 5 recommendations.

### 4. Start the API Service

To run the FastAPI production service:

```bash
python main.py
```

The API will be available at `http://localhost:8000`.

#### Endpoints:

- **GET `/health`**: Check service status.
- **POST `/recommend`**: Get recommendations for a user profile.

**Example Request Body:**

```json
{
  "user_profile": {
    "Biaya_Rata_Rata": 0,
    "Domain_Akademik": 0,
    "Domain_Bisnis_Karir": 1,
    "Domain_Olahraga_E-Sport": 0,
    "Domain_Seni_Kreatif": 0,
    "Domain_Teknologi": 1,
    "Domain_Umum_Lainnya": 0,
    "Jenjang_Encoded": 4,
    "Is_Online": 1,
    "Is_Offline": 0
  },
  "n": 5
}
```

## Persona Matching Logic

The model is trained on a synthetic dataset where users are matched with competitions based on their personas:

| Persona Name | Positive Match | Negative Match |
|--------------|----------------|----------------|
| Tech Mahasiswa Online | Teknologi, Bisnis/Karir | Seni Kreatif, Olahraga/E-Sport |
| Kreator Seni SMA Offline | Seni Kreatif | Teknologi, Bisnis/Karir |
| Atlet SMP/SMA | Olahraga/E-Sport | Akademik, Seni Kreatif |
| Akademisi Mahasiswa | Akademik | Olahraga/E-Sport, Seni Kreatif |
| Bisnis/Karir Umum | Bisnis/Karir | Seni Kreatif, Olahraga/E-Sport |

## Development

To add more personas or modify matching logic, update the `matching_rules` dictionary in `model_training.ipynb`.
