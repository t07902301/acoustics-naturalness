import librosa
from numpy.linalg import norm
import numpy as np
from dtw.dtw import dtw
import logging
import pydantic
from io import BytesIO
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ScoreResponse(pydantic.BaseModel):
    score: float = 0.0
    message: str = "Success"
    status: int = 200

def compute_discrepancy(query_stream: BytesIO, ref_stream: BytesIO) -> ScoreResponse:
    # Load query stream
    try:
        query_time_series, query_sample_rate = librosa.load(query_stream)
    except Exception as e:
        logger.error(f"Can't load the query audio: {e}")
        return ScoreResponse(
            message="the server can't load the query recording.", status=500
        )
    # Load reference stream
    try:
        ref_time_series, ref_sample_rate = librosa.load(ref_stream)  
    except Exception:
        logger.error("Can't load the reference audio")
        return ScoreResponse(
            message="the server can't load the reference audio", status=500
        )
    # Compute query's MFCC features and check if the audio is too long
    query_mfcc = librosa.feature.mfcc(
        y=query_time_series, sr=query_sample_rate, n_mfcc=13
    )
    try:
        assert query_mfcc.shape[1] < 1000
    except Exception:
        logger.error(f"query stream is too long: {query_mfcc.shape[1]}")
        return ScoreResponse(message="the user's recording is too long", status=500)
    try:
        assert query_mfcc.shape[1] > 1
    except Exception:
        logger.error(f"query stream is too short: {query_mfcc.shape[1]}")
        return ScoreResponse(message="the user's recording is too short", status=500)
    # Compute reference's MFCC features and check if the audio is too long
    ref_mfcc = librosa.feature.mfcc(y=ref_time_series, sr=ref_sample_rate, n_mfcc=13)
    try:
        assert ref_mfcc.shape[1] < 1000
    except Exception:
        logger.error(f"reference stream is too long: {ref_mfcc.shape[1]}")
        return ScoreResponse(message="the synthetic audio is too long", status=500)
    try:
        assert ref_mfcc.shape[1] > 1
    except Exception:
        logger.error(f"reference stream is too short: {ref_mfcc.shape[1]}")
        return ScoreResponse(message="the synthetic audio is too short", status=500)
    # Compute discrepancy score
    logger.info(f"query MFCC shape: {query_mfcc.shape}, ref MFCC shape: {ref_mfcc.shape}")
    score = np.round(compute_dtw(query_mfcc.T[:, 1:], ref_mfcc.T[:, 1:]),2)
    logger.info(f"discrepancy score: {score}")
    return ScoreResponse(score=score)

def compute_dtw(query: np.ndarray, ref: np.ndarray) -> float:
    """
    Compute the average cost of the optimal path w.r.t. the reference audio by Dynamic Time Warping (DTW).
    """
    dist, _, _, _ = dtw(query, ref, dist=lambda x, y: norm(x - y))
    avg_cost = dist / len(ref)
    return avg_cost