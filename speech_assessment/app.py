# -*- coding: utf-8 -*-
"""
@author: Gabriel Mittag, TU-Berlin
"""
import warnings
warnings.filterwarnings('ignore')
from nisqa.NISQA_model import nisqaModel
from flask import Flask, abort, request, jsonify
import logging
app = Flask(__name__)
from flask_cors import CORS
import os
from flask import json
from werkzeug.exceptions import HTTPException
cors = CORS(app, resources={r"/*": {"origins": "http://localhost:5000"}})
app.logger.setLevel(logging.INFO)
app.logger.info("Start the Server")

@app.route('/assess', methods=['POST'])
def predict_mos():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    audio = request.files['audio']
    app.logger.info(f"Receive {audio.filename}")

    audio_file = os.path.join('../file_buffer', audio.filename)
    audio.save(audio_file) # Save the FileStorage Object to the temporary file.
    app.logger.info(f"Save to {audio_file}")

    args = {
        "deg": audio_file,
        "mode": "predict_file",
        "pretrained_model": "weights/nisqa_tts.tar",
        "num_workers": 0,
        "bs": 1,
        "tr_bs_val": 1,
        "tr_num_workers": 0,
        "ms_channel": None,
        "data_dir": None,
        "output_dir": None,
        "csv_file": None,
        "csv_deg": None
    }
    try:
        nisqa = nisqaModel(args)
        results = nisqa.predict()
        return jsonify({'score': results["mos_pred"][0]})
    except Exception as e:
        abort(500, str(e))
    finally:
        os.remove(audio_file)
        app.logger.info(f"Remove {audio_file}")

@app.errorhandler(HTTPException)
def handle_exception(e):
    response = e.get_response()
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response

if __name__ == '__main__':
      app.run(host='0.0.0.0', port=6000)






















