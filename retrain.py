from pipeline import train_model
import json
import os
from config import STATUS_FILE


def set_training_status(status):

    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)

    with open(STATUS_FILE, "w") as f:
        json.dump({"training_status": status}, f)


def get_training_status():

    if not os.path.exists(STATUS_FILE):
        return "not_trained"

    try:
        with open(STATUS_FILE) as f:
            data = json.load(f)
            return data.get("training_status", "unknown")
    except:
        return "unknown"


def retrain_model():

    try:

        set_training_status("training")

        # jalankan training asli
        train_model()

        set_training_status("ready")

        return {
            "status": "success",
            "message": "training selesai"
        }

    except Exception as e:

        set_training_status("error")

        return {
            "status": "error",
            "message": str(e)
        }


if __name__ == "__main__":
    retrain_model()