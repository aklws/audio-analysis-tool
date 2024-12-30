from openai import OpenAI, APIError, AuthenticationError, RateLimitError
import uuid
import os
from loguru import logger
from dotenv import load_dotenv
from .audio_processor import AudioProcessor
from .feature_extractor import FeatureExtractor
from .visualizer import Visualizer

# 加载 .env 文件中的环境变量
load_dotenv()

class AudioEvaluator:
    def __init__(self):
        logger.info("初始化 AudioEvaluator")
        # 从环境变量中获取 API 密钥、基础 URL 和聊天模型
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        self.chat_model = os.getenv("OPENAI_CHAT_MODEL")

    def evaluate_audio(self, audio_path1, audio_path2):
        try:
            logger.info(f"开始分析音频: {audio_path1} 和 {audio_path2}")
            # 加载音频
            y1, sr1 = AudioProcessor.load_audio(audio_path1)
            y2, sr2 = AudioProcessor.load_audio(audio_path2)

            # 如果采样率不同，重采样
            if sr1 != sr2:
                logger.info(f"采样率不同: Audio 1 = {sr1} Hz, Audio 2 = {sr2} Hz。正在将 Audio 2 的采样率调整为 {sr1} Hz...")
                y2 = AudioProcessor.resample_audio(y2, orig_sr=sr2, target_sr=sr1)
                sr2 = sr1

            # 提取特征
            features1 = FeatureExtractor.extract_features(y1, sr1)
            features2 = FeatureExtractor.extract_features(y2, sr2)

            # 生成 UUID 子文件夹
            session_id = str(uuid.uuid4())
            session_dir = os.path.join("temp", session_id)
            os.makedirs(session_dir, exist_ok=True)

            # 可视化
            waveform_audio1_path = os.path.join(session_dir, "waveform_audio1.png")
            waveform_audio2_path = os.path.join(session_dir, "waveform_audio2.png")
            Visualizer.plot_waveform(y1, sr1, "Waveform of Audio 1", waveform_audio1_path)
            Visualizer.plot_waveform(y2, sr2, "Waveform of Audio 2", waveform_audio2_path)

            spectral_centroid_audio1_path = os.path.join(session_dir, "spectral_centroid_audio1.png")
            spectral_centroid_audio2_path = os.path.join(session_dir, "spectral_centroid_audio2.png")
            Visualizer.plot_spectral_centroid(y1, sr1, "Spectral Centroid of Audio 1", spectral_centroid_audio1_path)
            Visualizer.plot_spectral_centroid(y2, sr2, "Spectral Centroid of Audio 2", spectral_centroid_audio2_path)

            chromagram_audio1_path = os.path.join(session_dir, "chromagram_audio1.png")
            chromagram_audio2_path = os.path.join(session_dir, "chromagram_audio2.png")
            Visualizer.plot_chromagram(y1, sr1, "Chromagram of Audio 1", chromagram_audio1_path)
            Visualizer.plot_chromagram(y2, sr2, "Chromagram of Audio 2", chromagram_audio2_path)

            spectrogram_comparison_path = os.path.join(session_dir, "spectrogram_comparison.png")
            Visualizer.plot_spectrogram_comparison(y1, sr1, y2, sr2, "Spectrogram Comparison", spectrogram_comparison_path)

            # 组织特征描述
            prompt = (
                "以下是两段音频的特征描述，请判断它们的音色是否相近，以及克隆效果是否准确。注意，两段音频的时长可能不同：\n\n"
                f"音频 1 时长: {len(y1) / sr1:.2f} 秒\n"
                f"音频 2 时长: {len(y2) / sr2:.2f} 秒\n\n"
                "音频 1 特征：\n"
                f"{features1}\n"
                "音频 2 特征：\n"
                f"{features2}\n\n"
                "请分析并给出结论，同时给出相似度打分（满分 100%）。"
            )

            # 调用 OpenAI API
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "你是一个音频分析专家，能够根据音频特征判断音色相似度和克隆效果。"},
                    {"role": "user", "content": prompt}
                ],
            )

            # 获取 OpenAI 的回复
            ai_response = response.choices[0].message.content

            # 询问如何改进克隆效果
            improvement_suggestion = self.ask_for_improvement(ai_response)

            # 返回结果
            result = {
                "AI 分析结果": ai_response,
                "改进建议": improvement_suggestion,
                "音频 1 特征": features1,
                "音频 2 特征": features2,
                "波形图 Audio 1": waveform_audio1_path,
                "波形图 Audio 2": waveform_audio2_path,
                "频谱质心图 Audio 1": spectral_centroid_audio1_path,
                "频谱质心图 Audio 2": spectral_centroid_audio2_path,
                "色度图 Audio 1": chromagram_audio1_path,
                "色度图 Audio 2": chromagram_audio2_path,
                "声谱对比图": spectrogram_comparison_path
            }
            return result
        except (APIError, AuthenticationError, RateLimitError) as e:
            # 捕获 OpenAI API 相关的异常
            logger.error(f"OpenAI API 调用失败: {str(e)}")
            return {"error": f"OpenAI API 调用失败: {str(e)}"}
        except Exception as e:
            # 捕获其他异常
            logger.error(f"分析音频时发生错误: {str(e)}")
            return {"error": f"处理音频时发生错误: {str(e)}"}


    def ask_for_improvement(self, analysis_result):
        """询问如何改进克隆效果"""
        prompt = (
            "以下是对两段音频的分析结果：\n\n"
            f"{analysis_result}\n\n"
            "请根据分析结果，给出改进克隆效果的具体建议。"
        )

        # 调用 OpenAI API
        response = self.client.chat.completions.create(
            model=self.chat_model,  # 使用 self.chat_model
            messages=[
                {"role": "system", "content": "你是一个音频克隆专家，能够根据分析结果提出改进克隆效果的建议。"},
                {"role": "user", "content": prompt}
            ],
        )

        # 获取 OpenAI 的回复
        return response.choices[0].message.content