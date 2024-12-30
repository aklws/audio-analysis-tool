import librosa
from loguru import logger

class AudioProcessor:
    @staticmethod
    def load_audio(file_path, sr=None):
        """加载音频文件"""
        logger.info("加载音频文件")
        return librosa.load(file_path, sr=sr)

    @staticmethod
    def resample_audio(y, orig_sr, target_sr):
        """重采样音频"""
        logger.info("重采样音频")
        return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)