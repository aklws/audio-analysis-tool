import gradio as gr
from audio_analysis.evaluator import AudioEvaluator
from loguru import logger

# 配置日志
logger.add("audio_analysis.log", rotation="10 MB") 

# 初始化评估器
evaluator = AudioEvaluator()

# 定义 Gradio 界面
def audio_evaluation_interface(audio_file1, audio_file2):
    try:
        result = evaluator.evaluate_audio(audio_file1, audio_file2)
        result_str = (
            f"AI 分析结果:\n{result['AI 分析结果']}\n\n"
            f"改进建议:\n{result['改进建议']}\n\n"
            f"音频 1 特征:\n{result['音频 1 特征']}\n\n"
            f"音频 2 特征:\n{result['音频 2 特征']}"
        )
        return (
            result_str,
            result["波形图 Audio 1"],
            result["波形图 Audio 2"],
            result["频谱质心图 Audio 1"],
            result["频谱质心图 Audio 2"],
            result["色度图 Audio 1"],
            result["色度图 Audio 2"],
            result["声谱对比图"]
        )
    except Exception as e:
        logger.error(f"询问ai时发生错误: {str(e)}")
        return {}

# 使用 gr.Blocks 创建应用
with gr.Blocks() as app:
    gr.Markdown("# 音频音色对比与克隆效果分析工具")
    gr.Markdown("上传两个音频文件，分析其音色相似度和克隆效果。")
    with gr.Row():
        audio_input1 = gr.Audio(type="filepath", label="上传音频文件 1")
        audio_input2 = gr.Audio(type="filepath", label="上传音频文件 2")
    with gr.Row():
        result_output = gr.Markdown(label="分析结果")
    with gr.Row():
        waveform_output1 = gr.Image(label="波形图 - Audio 1")
        waveform_output2 = gr.Image(label="波形图 - Audio 2")
    with gr.Row():
        spectral_centroid_output1 = gr.Image(label="频谱质心图 - Audio 1")
        spectral_centroid_output2 = gr.Image(label="频谱质心图 - Audio 2")
    with gr.Row():
        chromagram_output1 = gr.Image(label="色度图 - Audio 1")
        chromagram_output2 = gr.Image(label="色度图 - Audio 2")
    with gr.Row():
        spectrogram_comparison_output = gr.Image(label="声谱对比图")
    submit_button = gr.Button("分析")
    submit_button.click(
        audio_evaluation_interface,
        inputs=[audio_input1, audio_input2],
        outputs=[
            result_output,
            waveform_output1,
            waveform_output2,
            spectral_centroid_output1,
            spectral_centroid_output2,
            chromagram_output1,
            chromagram_output2,
            spectrogram_comparison_output
        ]
    )

# 启动应用
app.launch(max_file_size=None)  # 去掉文件大小限制