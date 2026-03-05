from flask import Flask,request,jsonify
from inference import recommend
from pipeline import train_model
from retrain import retrain_model, get_training_status

app = Flask(__name__)


@app.route("/")
def home():
    return "GoTrip Recommendation API"

## Status

@app.route("/status", methods=["GET"])
def status():

    return jsonify({
        "status": "success",
        "training_status": get_training_status()
    })


############################################################
# INFERENCE MODE
############################################################

@app.route("/inference",methods=["POST"])
def inference():

    data = request.get_json()

    user_id = data.get("user_id")
    top_n = int(data.get("top_n",5))

    result = recommend(user_id,top_n)

    return jsonify({
        "status":"success",
        "data":result
    })


############################################################
# RETRAIN MODE
############################################################

@app.route("/retrain",methods=["POST"])
def retrain():

    train_model()

    return jsonify({
        "status":"success",
        "message":"model retrained"
    })


if __name__ == "__main__":
    app.run()