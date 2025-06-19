# 🤖 RCH-StackBot-3.8B

**Swedish AI Assistant for Fullstack Development**

An intelligent chatbot based on Microsoft Phi-3 Mini that helps Swedish web developers with React, Node.js, JavaScript, CSS and more.

![Status](https://img.shields.io/badge/Status-Working-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

- 🇸🇪 **Swedish Language** - Natural conversation in Swedish
- 🚀 **Fullstack Focus** - React, Node.js, JavaScript, CSS, databases
- 💻 **Multiple Interfaces** - Command line, chat, web app
- ⚡ **GPU Acceleration** - Faster responses with CUDA
- 🎨 **Modern Web Interface** - Streamlit with dark/light theme

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/RCH-StackBot-3.8B.git
cd RCH-StackBot-3.8B

# 2. Create virtual environment
python -m venv phi3_env
phi3_env\Scripts\activate  # Windows
# source phi3_env/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start chat
python inference.py --interactive
```

## 💻 Usage

### Command Line

```bash
# Single question
python inference.py --prompt "How do I create a React component?"

# Interactive chat
python inference.py --interactive

# With custom settings
python inference.py --prompt "Explain async/await" --temperature 0.5
```

### Web Interface

```bash
streamlit run streamlit_app.py
```

Then open `http://localhost:8501`

### Advanced Interface

```bash
python stackbot.py --interactive
```

## 🎯 Examples

**Questions you can ask (in Swedish):**

- "Vad är React och hur använder jag det?" (What is React and how do I use it?)
- "Hur skapar jag en Express server?" (How do I create an Express server?)
- "Förklara skillnaden mellan let och const" (Explain the difference between let and const)
- "Visa kod för en fetch-request" (Show code for a fetch request)
- "Vad är CORS och hur fixar jag det?" (What is CORS and how do I fix it?)

## ⚙️ System Requirements

**Minimum:**

- Python 3.8+
- 16GB RAM
- 10GB disk space

**Recommended:**

- NVIDIA GPU with 8GB+ VRAM
- 32GB RAM

## 🔧 Training (Optional)

Add Swedish Q&A pairs to `data/training_data.txt`:

```
Fråga: Hur skapar jag en React komponent?
Svar: För att skapa en React komponent...
```

Start training:

```bash
python train.py --epochs 3 --batch_size 4
```

## 🐛 Common Issues

**Model not loading:**

```bash
python scripts/setup.py --force-download
```

**CUDA memory error:**

```bash
python inference.py --prompt "test" --device cpu
```

**Check GPU:**

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 📁 Project Structure

```
RCH-StackBot-3.8B/
├── 🐍 inference.py          # Main interface
├── 🐍 train.py              # Training script
├── 🐍 stackbot.py           # Advanced interface
├── 🐍 streamlit_app.py      # Web app
├── 📄 requirements.txt      # Dependencies
├── 📁 data/                 # Training data
├── 📁 models/               # AI models
└── 📁 scripts/              # Helper scripts
```

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Microsoft for Phi-3 Mini
- Hugging Face for Transformers
- Streamlit for web interface

---
