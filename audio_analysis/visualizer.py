import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from loguru import logger
class Visualizer:
    @staticmethod
    def plot_waveform(y, sr, title, filename):
        """绘制波形图"""
        logger.info("绘制波形图")
        plt.figure(figsize=(10, 4))
        times = librosa.times_like(y, sr=sr)
        plt.plot(times, y)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.xlim(0, max(times))
        plt.tight_layout()
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_spectral_centroid(y, sr, title, filename):
        """绘制频谱质心图"""
        logger.info("绘制频谱质心图")
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        times = librosa.times_like(spectral_centroids, sr=sr)

        plt.figure(figsize=(10, 4))
        plt.plot(times, spectral_centroids, label="Spectral Centroid")
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_chromagram(y, sr, title, filename):
        """绘制色度图"""
        logger.info("绘制色度图")
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_spectrogram_comparison(y1, sr1, y2, sr2, title, filename):
        """绘制声谱对比图"""
        logger.info("绘制声谱对比图")
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y1)), ref=np.max), sr=sr1, x_axis='time', y_axis='log')
        plt.title("Spectrogram of Audio 1")
        plt.subplot(2, 1, 2)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y2)), ref=np.max), sr=sr2, x_axis='time', y_axis='log')
        plt.title("Spectrogram of Audio 2")
        plt.tight_layout()
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()