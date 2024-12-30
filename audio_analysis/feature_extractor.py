import librosa
import numpy as np
from loguru import logger

class FeatureExtractor:
    @staticmethod
    def extract_features(y, sr):
        """提取音频特征"""
        try:
            # MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            logger.info("提取MFCC")

            # Chroma
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            logger.info("提取Chroma")

            # Spectral Contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
            logger.info("提取Spectral Contrast")

            # Spectral Rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_rolloff_mean = np.mean(spectral_rolloff, axis=1)
            logger.info("提取Spectral Rolloff")

            # Zero-Crossing Rate
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            zero_crossing_rate_mean = np.mean(zero_crossing_rate)
            logger.info("提取Zero-Crossing Rate")

            # RMS Energy
            rms_energy = librosa.feature.rms(y=y)
            rms_energy_mean = np.mean(rms_energy)
            logger.info("提取RMS Energy")

            # Spectral Bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_bandwidth_mean = np.mean(spectral_bandwidth)
            logger.info("提取Spectral Bandwidth")

            # Spectral Centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_centroid_mean = np.mean(spectral_centroid)
            logger.info("提取Spectral Centroid")

            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            logger.info("提取Tempo")

            # 返回特征描述
            features = {
                "mfcc_mean": mfcc_mean,
                "chroma_mean": chroma_mean,
                "spectral_contrast_mean": spectral_contrast_mean,
                "spectral_rolloff_mean": spectral_rolloff_mean,
                "zero_crossing_rate_mean": zero_crossing_rate_mean,
                "rms_energy_mean": rms_energy_mean,
                "spectral_bandwidth_mean": spectral_bandwidth_mean,
                "spectral_centroid_mean": spectral_centroid_mean,
                "tempo": tempo,
            }
            return features
        except Exception as e:
            logger.error(f"提取音频特征时发生错误: {str(e)}")
            return {}