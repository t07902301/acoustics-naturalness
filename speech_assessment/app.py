# -*- coding: utf-8 -*-
"""
@author: Gabriel Mittag, TU-Berlin
"""

import warnings
from flask_cors import CORS
import os
from flask import json
from werkzeug.exceptions import HTTPException
from audio_judge import compute_similarity
from flask import Flask, abort, request, jsonify
import logging
from werkzeug.datastructures import FileStorage

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
    response = e.get_response()
    response.data = json.dumps(
        {
            "code": e.code,
            "name": e.name,
            "description": e.description,
        }
    )
    response.content_type = "application/json"
    return response

@app.route("/api/similarity_scores", methods=["POST"])
def get_similarity_scores():
    query_audio = request.files["query_audio"]
    ref_audio = request.files["reference_audio"]
    query_audio_path = generate_audio_file(query_audio, "query")
    ref_audio_path = generate_audio_file(ref_audio, "ref")
    try:
        response = compute_similarity(query_audio_path, ref_audio_path)
    except Exception as e:
        app.logger.error(e)
        abort(500, str(e))
    finally:
        os.remove(query_audio_path)
        os.remove(ref_audio_path)
    if response.status != 200:
        abort(response.status, response.message)
    return jsonify({"score": response.score})


@app.route("/health", methods=["GET"])
def healthcheck():
    try:
        app.logger.info("Health Checking")
        response = compute_similarity(
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
