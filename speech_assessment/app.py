# -*- coding: utf-8 -*-
"""
@author: Gabriel Mittag, TU-Berlin
"""

import warnings
from flask_cors import CORS
import os
from werkzeug.exceptions import HTTPException
from utils import compute_discrepancy
from flask import Flask, abort, request, jsonify
import logging
from werkzeug.datastructures import FileStorage
from io import BytesIO
warnings.filterwarnings("ignore")

app = Flask(__name__)

cors = CORS(app, resources={r"/*": {"origins": ["http://tutor-dev-backend-1:5000", "http://tutor-staging-backend-1:5000", "http://tutor-prod-backend-1:5000"]}})
app.logger.setLevel(logging.INFO)
app.logger.info("Start the Server")

def generate_audio_file(audio: FileStorage, name: str) -> str:
    audio_path = os.path.join("file_buffer", name + ".wav")
    audio.save(audio_path)
    return audio_path


@app.errorhandler(HTTPException)
def handle_exception(e):
    # 1. Create your JSON payload
    payload = {
        'message': e.description, 
        'code': e.code
    }
    
    # 2. Return a tuple: (JSON response, Status Code integer)
    # This forces Flask to update the actual network status header!
    return jsonify(payload), e.code


@app.route("/api/discrepancy_score", methods=["POST"])
def get_discrepancy_score():
    query_audio = request.files["query_audio"]
    ref_audio = request.files["reference_audio"]
    query_stream = BytesIO(query_audio.read())
    query_stream.seek(0)
    ref_stream = BytesIO(ref_audio.read())
    ref_stream.seek(0)
    try:
        response = compute_discrepancy(query_stream, ref_stream)
        if response.status != 200:
            abort(response.status, response.message)
        return jsonify({"score": response.score})
    except Exception as e:
        app.logger.error(e)
        abort(500, str(e))


@app.route("/health", methods=["GET"])
def healthcheck():
    try:
        app.logger.info("Health Checking")
        response = compute_discrepancy(
            os.path.join("file_buffer", "bad.wav"),
            os.path.join("file_buffer", "generated.wav"),
        )
    except Exception as e:
        app.logger.error(e)
        abort(500, str(e))
    if response.status != 200:
        abort(response.status, response.message)
    return jsonify({"score": response.score})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)
