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
from pydub import AudioSegment
import io

warnings.filterwarnings("ignore")

app = Flask(__name__)

cors = CORS(app, resources={r"/*": {"origins": ["http://tutor-dev-backend-1:5000", "http://tutor-staging-backend-1:5000", "http://tutor-prod-backend-1:5000"]}})
app.logger.setLevel(logging.INFO)
app.logger.info("Start the Server")



def load_fileStorage(audio: FileStorage, name: str) -> str:
    current_path = os.path.dirname(os.path.abspath(__file__))
    audio_path = os.path.join(current_path, "file_buffer", f"{name}.wav")
    
    # Read the file data into memory
    file_bytes = audio.read()
    
    # Load the audio data regardless of the input format
    # This automatically detects if it's webm, ogg, wav, etc.
    audio_segment = AudioSegment.from_file(io.BytesIO(file_bytes))
    
    # Export as a standardized WAV
    audio_segment.export(audio_path, format="wav")
    
    return audio_path

@app.errorhandler(HTTPException)
def handle_exception(e):
    response = jsonify({'message': e.description, 'code': e.code})
    return response

@app.route("/api/discrepancy_score", methods=["POST"])
def get_discrepancy_score():
    query_audio = request.files["query_audio"]
    ref_audio = request.files["reference_audio"]
    query_audio_path = load_fileStorage(query_audio, "query")
    ref_audio_path = load_fileStorage(ref_audio, "ref")
    try:
        response = compute_discrepancy(query_audio_path, ref_audio_path)
    except Exception as e:
        app.logger.error(e)
        abort(500, str(e))
    # finally:
    #     os.remove(query_audio_path)
    #     os.remove(ref_audio_path)
    if response.status != 200:
        abort(response.status, response.message)
    return jsonify({"score": response.score})


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
