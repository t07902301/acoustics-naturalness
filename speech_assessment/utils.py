import librosa
from numpy.linalg import norm
import numpy as np
from dtw.dtw import dtw
import logging
import pydantic

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class ScoreResponse(pydantic.BaseModel):
    score: float = 0.0
    message: str = "Success"
    status: int = 200

def compute_discrepancy(query_audio_file: str, ref_audio_file: str, query_span: dict = None, ref_span: dict = None) -> ScoreResponse:
    # Load full files
    y_ref, sr = librosa.load(ref_audio_file, sr=16000)
    y_query, _ = librosa.load(query_audio_file, sr=16000)

    # Apply offset in memory (More precise than FFmpeg)
    y_query = y_query[int(query_span.get('start', 0) * sr):] if query_span is not None else y_query
    y_ref = y_ref[int(ref_span.get('start', 0) * sr): int(ref_span.get('end', len(y_ref) / sr) * sr)] if ref_span is not None else y_ref

    # 4. Extract MFCCs
    mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr)
    mfcc_query = librosa.feature.mfcc(y=y_query, sr=sr)

    log.info(f"MFCC shapes: ref {mfcc_ref.shape}, query {mfcc_query.shape}")
        
    # # 5. Z-Score Normalize (This fixes the Energy/Volume Mismatch)
    # mfcc_ref = (mfcc_ref - np.mean(mfcc_ref)) / np.std(mfcc_ref)
    # mfcc_query = (mfcc_query - np.mean(mfcc_query)) / np.std(mfcc_query)

    score = np.round(compute_dtw(mfcc_query.T[:, 1:], mfcc_ref.T[:, 1:]),2)
    return ScoreResponse(score=score)

    # # Load query audio file
    # try:
    #     query_time_series, query_sample_rate = librosa.load(query_audio_file)
    # except Exception:
    #     log.error(f"Can't load {query_audio_file}")
    #     return ScoreResponse(
    #         message="the server can't load the user's recording.", status=500
    #     )
    # # Load reference audio file
    # try:
    #     ref_time_series, ref_sample_rate = librosa.load(ref_audio_file)
    # except Exception:
    #     log.error(f"Can't load {ref_audio_file}")
    #     return ScoreResponse(
    #         message="the server can't load synthetic audio", status=500
    #     )
    
    # # Compute query's MFCC features and check if the audio is too long
    # query_mfcc = librosa.feature.mfcc(
    #     y=query_time_series, sr=query_sample_rate, n_mfcc=13
    # )
    # try:
    #     assert query_mfcc.shape[1] < 1000
    # except Exception:
    #     log.error(f"{query_audio_file} too long: {query_mfcc.shape[1]}")
    #     return ScoreResponse(message="the user's recording is too long", status=500)
    # try:
    #     assert query_mfcc.shape[1] > 1
    # except Exception:
    #     log.error(f"{query_audio_file} too short: {query_mfcc.shape[1]}")
    #     return ScoreResponse(message="the user's recording is too short", status=500)
    # # Compute reference's MFCC features and check if the audio is too long
    # ref_mfcc = librosa.feature.mfcc(y=ref_time_series, sr=ref_sample_rate, n_mfcc=13)
    # try:
    #     assert ref_mfcc.shape[1] < 1000
    # except Exception:
    #     log.error(f"{ref_audio_file} too long: {ref_mfcc.shape[1]}")
    #     return ScoreResponse(message="the synthetic audio is too long", status=500)
    # try:
    #     assert ref_mfcc.shape[1] > 1
    # except Exception:
    #     log.error(f"{ref_audio_file} too short: {ref_mfcc.shape[1]}")
    #     return ScoreResponse(message="the synthetic audio is too short", status=500)
    # # Compute discrepancy score
    # score = np.round(compute_dtw(query_mfcc.T[:, 1:], ref_mfcc.T[:, 1:]),2)
    # return ScoreResponse(score=score)

def compute_dtw(query: np.ndarray, ref: np.ndarray) -> float:
    """
    Compute the average cost of the optimal path when aligning two sequences by Dynamic Time Warping (DTW).
    """
    dist, _, _, path = dtw(query, ref, dist=lambda x, y: norm(x - y))
    avg_cost = dist / len(path[0])
    return avg_cost