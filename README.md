[简体中文](README_zh.md)
## Audio Tone Comparison and Clone Effects Analysis Tool

## Project Information
This project is an audio tone comparison and clone effect analysis tool. It extracts audio features such as MFCC, tempo, spectral center of mass, etc., and uses OpenAI's API to analyze the similarity and cloning effect of two pieces of audio. The project also provides visualization features, including waveform graphs, spectral plasmas, chromaticity graphs, and sound spectrum comparison graphs. It is used to judge the quality of sound clones when you cannot hear them by ear.

## Functional Features
- **Audio Feature Extraction**: Extract features such as MFCC, tempo, spectral center of mass, spectral bandwidth, etc.
- **Tone Similarity Analysis**: Use OpenAI's API to analyze the tone similarity and cloning effect of two pieces of audio.
- **Improvement Suggestion**: Based on the analysis results, provide specific suggestions to improve the cloning effect.
- **Visualization Functions**: Generate waveform map, spectral plasma map, chromaticity map and sound spectrum comparison map.
- **User-friendly interface**: Interactive Gradio-based interface that supports uploading audio files and viewing analysis results.

## Installation steps
1. **Clone the project**
   ```bash
   git clone https://github.com/aklws/audio-analysis-tool.git
   cd audio-analysis-tool

2. **Install dependencies** ``bash
    ```bash
    pip install -r requirements.txt

3. **Configure Environment Variables**
    Create a .env file in the root directory of the project and fill in the following:
    ```bash
    OPENAI_API_KEY=your_openai_api_key
    OPENAI_BASE_URL=your_base_url
    OPENAI_CHAT_MODEL=your_base_model

## How to use
1. **Start the application**
    ```bash
    python app.py

2. **Accessing the Gradio interface** **Accessing the Gradio interface** **Accessing the Gradio interface
    - Open your browser and visit http://127.0.0.1:7860, you will see the following interface:
    - Upload Audio Files: Upload two audio files (WAV, MP3, etc. formats supported).
    - Analysis Results: View OpenAI's analysis results, similarity scores, and suggestions for improvement.
    - Visual Graphs: View waveform graphs, spectral plots, chromaticity graphs, and sound spectrum comparison graphs.

3. **Analyze Audio**
    - Upload two audio files. On the left is the reference audio and on the right is the cloned audio, both of which say the same thing.
    - Click the “Analyze” button and wait for the analysis to complete.
    - View the analysis results and visualization charts.

## License
    This project is licensed under the MIT license. See the LICENSE file for details.

Translated with DeepL.com (free version)