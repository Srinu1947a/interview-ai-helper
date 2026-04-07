<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/AI-Multi--Provider-purple?style=for-the-badge" alt="AI">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=for-the-badge" alt="PRs Welcome">
</p>

<h1 align="center">🎤 Interview Helper</h1>

<p align="center">
  <strong>Real-time AI-powered interview assistant that listens, translates, and generates answer suggestions — all in your browser.</strong>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-demo">Demo</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#%EF%B8%8F-configuration">Configuration</a> •
  <a href="#-keyboard-shortcuts">Shortcuts</a> •
  <a href="#-supported-languages">Languages</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## ✨ Features

- **Real-time Speech Recognition** — Uses the browser's Web Speech API for instant transcription
- **15+ Languages** — Input and target language freely selectable (English, German, French, Spanish, Chinese, Japanese, and more)
- **AI Answer Suggestions** — Professional, confident answer suggestions in both languages
- **10 AI Providers** — Qwen, OpenAI, Anthropic, Google Gemini, Mistral, Groq, DeepSeek, Together AI, OpenRouter, Ollama (local/free)
- **Interview / Conversation Mode** — Switch between full interview mode (with answer suggestions) and simple conversation mode (translation only)
- **Dark/Light Mode** — Easy on the eyes during long sessions
- **Keyboard Shortcuts** — Control everything without touching the mouse
- **Export** — Download your entire interview transcript as Markdown
- **Sound Feedback** — Subtle audio cues when processing starts and completes
- **Settings Panel** — Configure API keys, models, and base URLs directly in the UI
- **Zero Dependencies Frontend** — Single Python file, no build step, no npm

## 🎬 Demo

<!-- Replace with actual screenshot/GIF -->
<p align="center">
  <img src="https://via.placeholder.com/800x450/181a24/6c5ce7?text=🎤+Interview+Helper+Demo" alt="Interview Helper Demo" width="800">
</p>

> **How it works:** Start listening → speak or let your interviewer speak → the tool transcribes in real-time → detects pauses → translates the question → generates answer suggestions in both languages.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- A modern browser (Chrome or Edge recommended for best speech recognition)
- An API key from one of the supported providers

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/interview-helper.git
cd interview-helper

# Install dependencies
pip install -r requirements.txt

# Set your API key (choose one)
export QWEN_API_KEY="your-key-here"        # Qwen / DashScope
export OPENAI_API_KEY="your-key-here"       # OpenAI
export ANTHROPIC_API_KEY="your-key-here"    # Anthropic
export GEMINI_API_KEY="your-key-here"       # Google Gemini
export GROQ_API_KEY="your-key-here"         # Groq (fast inference)
# ... or configure in the UI via ⚙️ Settings

# Start the server
python interview_helper.py
```

**Windows (PowerShell):**
```powershell
$env:QWEN_API_KEY = "your-key-here"
python interview_helper.py
```

Then open **http://localhost:5000** in your browser.

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `QWEN_API_KEY` | Qwen / DashScope API key | — |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `GEMINI_API_KEY` | Google Gemini API key | — |
| `MISTRAL_API_KEY` | Mistral AI API key | — |
| `GROQ_API_KEY` | Groq API key | — |
| `DEEPSEEK_API_KEY` | DeepSeek API key | — |
| `TOGETHER_API_KEY` | Together AI API key | — |
| `OPENROUTER_API_KEY` | OpenRouter API key | — |
| `PORT` | Server port | `5000` |

Each provider also supports `*_MODEL` and `*_BASE_URL` env vars (e.g. `QWEN_MODEL`, `OPENAI_BASE_URL`).

### In-App Settings

Click **⚙️ Settings** (or press `Ctrl+,`) to configure API keys, models, and base URLs without restarting.

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Start / Stop listening |
| `Escape` | Clear history / Close settings |
| `Ctrl+E` | Export transcript as Markdown |
| `Ctrl+,` | Open settings |
| `T` | Toggle dark/light mode |
| `M` | Toggle interview/conversation mode |

## 🌍 Supported Languages

| Language | Code | Input | Target |
|----------|------|:-----:|:------:|
| English | en | ✅ | ✅ |
| Deutsch | de | ✅ | ✅ |
| Français | fr | ✅ | ✅ |
| Español | es | ✅ | ✅ |
| 中文 | zh | ✅ | ✅ |
| 日本語 | ja | ✅ | ✅ |
| 한국어 | ko | ✅ | ✅ |
| Português | pt | ✅ | ✅ |
| Italiano | it | ✅ | ✅ |
| Русский | ru | ✅ | ✅ |
| العربية | ar | ✅ | ✅ |
| Türkçe | tr | ✅ | ✅ |
| Polski | pl | ✅ | ✅ |
| Nederlands | nl | ✅ | ✅ |
| हिन्दी | hi | ✅ | ✅ |

## 🏗️ Tech Stack

- **Backend:** Python, Flask, Flask-SocketIO
- **Frontend:** Vanilla JS, Web Speech API, Socket.IO
- **AI:** OpenAI-compatible API (works with any provider)
- **Zero build step** — just `python interview_helper.py`

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Ideas for contributions

- [ ] Add more AI providers (Google Gemini, Mistral, etc.)
- [ ] Whisper API integration for server-side transcription
- [ ] Custom system prompts / interview context
- [ ] Save/load interview sessions
- [ ] Mobile-optimized layout
- [ ] Browser extension version

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## ⭐ Star History

If this tool helped you ace your interview, consider giving it a star!

---

<p align="center">
  Made with ❤️ for job seekers everywhere
</p>
