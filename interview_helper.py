#!/usr/bin/env python3
"""
Interview Helper — Real-time Interview Translation & Answer Assistant
=====================================================================
Listens to your interview via browser microphone, transcribes speech,
translates questions, and generates professional answer suggestions
using AI — all in real-time.

Supports: OpenAI, Anthropic, Qwen (DashScope), Ollama (local)

Start:  python interview_helper.py
Open:   http://localhost:5000
"""

import os
import json
import re
import threading
from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit

# ── Configuration ──────────────────────────────────────────────────────────
DEFAULT_PORT = int(os.environ.get("PORT", 5000))

PROVIDERS = {
    "qwen": {
        "name": "Qwen (DashScope)",
        "base_url": os.environ.get("QWEN_BASE_URL", "https://coding-intl.dashscope.aliyuncs.com/v1"),
        "api_key": os.environ.get("QWEN_API_KEY", ""),
        "model": os.environ.get("QWEN_MODEL", "qwen3.5-plus"),
    },
    "openai": {
        "name": "OpenAI",
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o"),
    },
    "anthropic": {
        "name": "Anthropic",
        "base_url": None,
        "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
    },
    "gemini": {
        "name": "Google Gemini",
        "base_url": os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"),
        "api_key": os.environ.get("GEMINI_API_KEY", ""),
        "model": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    },
    "mistral": {
        "name": "Mistral AI",
        "base_url": os.environ.get("MISTRAL_BASE_URL", "https://api.mistral.ai/v1"),
        "api_key": os.environ.get("MISTRAL_API_KEY", ""),
        "model": os.environ.get("MISTRAL_MODEL", "mistral-large-latest"),
    },
    "groq": {
        "name": "Groq (Fast)",
        "base_url": os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        "api_key": os.environ.get("GROQ_API_KEY", ""),
        "model": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
    },
    "together": {
        "name": "Together AI",
        "base_url": os.environ.get("TOGETHER_BASE_URL", "https://api.together.xyz/v1"),
        "api_key": os.environ.get("TOGETHER_API_KEY", ""),
        "model": os.environ.get("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    },
    "openrouter": {
        "name": "OpenRouter",
        "base_url": os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
        "model": os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash"),
    },
    "ollama": {
        "name": "Ollama (Local)",
        "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "api_key": "ollama",
        "model": os.environ.get("OLLAMA_MODEL", "llama3.1"),
    },
}

LANGUAGES = {
    "de": "Deutsch", "en": "English", "fr": "Français", "es": "Español",
    "zh": "中文", "ja": "日本語", "ko": "한국어", "pt": "Português",
    "it": "Italiano", "ru": "Русский", "ar": "العربية", "tr": "Türkçe",
    "pl": "Polski", "nl": "Nederlands", "hi": "हिन्दी",
}

SPEECH_LANG_CODES = {
    "de": "de-DE", "en": "en-US", "fr": "fr-FR", "es": "es-ES",
    "zh": "zh-CN", "ja": "ja-JP", "ko": "ko-KR", "pt": "pt-BR",
    "it": "it-IT", "ru": "ru-RU", "ar": "ar-SA", "tr": "tr-TR",
    "pl": "pl-PL", "nl": "nl-NL", "hi": "hi-IN",
}

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24).hex()
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── State ──────────────────────────────────────────────────────────────────
conversation_history = []
msg_counter = 0

# User-configurable interview settings (updated via UI)
interview_settings = {
    "context": "",          # e.g. "Senior Developer at Google, 5 years experience"
    "answer_length": "medium",  # short / medium / long
    "answer_tone": "professional",  # professional / casual / technical
    "custom_prompt": "",    # additional instructions
}


def get_openai_client(provider_key):
    """Create an OpenAI-compatible client for the given provider."""
    from openai import OpenAI
    cfg = PROVIDERS[provider_key]
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])


def get_anthropic_client():
    """Create an Anthropic client."""
    import anthropic
    return anthropic.Anthropic(api_key=PROVIDERS["anthropic"]["api_key"])


def call_ai(prompt, provider_key):
    """Unified AI call across all providers."""
    cfg = PROVIDERS[provider_key]

    if provider_key == "anthropic":
        client = get_anthropic_client()
        response = client.messages.create(
            model=cfg["model"],
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    else:
        client = get_openai_client(provider_key)
        response = client.chat.completions.create(
            model=cfg["model"],
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content.strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text


# ── HTML Template ──────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Interview Helper</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🎤</text></svg>">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #07080d;
    --bg2: #0e1018;
    --card: #151720;
    --border: #1f2233;
    --border-hover: #2d3148;
    --accent: #7c6cf0;
    --accent2: #a8a0ff;
    --accent-glow: rgba(124,108,240,.15);
    --text: #eaecf2;
    --text2: #c8cad4;
    --muted: #5c6078;
    --green: #2dd4bf;
    --green-bg: rgba(45,212,191,.06);
    --orange: #ffb347;
    --red: #f87171;
    --blue: #60a5fa;
    --radius: 14px;
    --radius-sm: 8px;
    --shadow: 0 2px 16px rgba(0,0,0,.25);
    --shadow-lg: 0 8px 40px rgba(0,0,0,.4);
    --font-size: 14px;
    --transition: .2s cubic-bezier(.16,1,.3,1);
  }
  [data-theme="light"] {
    --bg: #f4f5f9;
    --bg2: #eaecf2;
    --card: #ffffff;
    --border: #d8dae4;
    --border-hover: #c0c4d4;
    --text: #1a1c2e;
    --text2: #3a3d52;
    --muted: #8b8fa3;
    --accent-glow: rgba(124,108,240,.08);
    --green-bg: rgba(45,212,191,.05);
    --shadow: 0 2px 12px rgba(0,0,0,.06);
    --shadow-lg: 0 8px 40px rgba(0,0,0,.1);
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    transition: background .3s, color .3s;
    font-size: var(--font-size);
    -webkit-font-smoothing: antialiased;
  }

  /* ── Header ── */
  .header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 24px;
    background: var(--card);
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 100;
  }
  .brand { display: flex; align-items: center; gap: 10px; }
  .brand h1 {
    font-size: 18px; font-weight: 700;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .brand-badge {
    font-size: 9px; padding: 2px 8px; border-radius: 20px;
    background: var(--accent-glow); color: var(--accent); font-weight: 600;
    border: 1px solid rgba(124,108,240,.3);
    letter-spacing: .5px; text-transform: uppercase;
  }
  .header-right { display: flex; align-items: center; gap: 12px; }
  .header-stats {
    display: flex; gap: 16px; font-size: 12px; color: var(--muted);
  }
  .header-stats span { display: flex; align-items: center; gap: 5px; }
  .status-bar { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--muted); }
  .status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--red); transition: background .3s;
  }
  .status-dot.listening { background: var(--green); animation: pulse 1.5s infinite; }
  @keyframes pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(45,212,191,.4); }
    50% { box-shadow: 0 0 0 8px rgba(45,212,191,0); }
  }

  /* ── Controls Bar ── */
  .controls {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 24px;
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
  }
  .controls-left { display: flex; gap: 6px; flex: 1; flex-wrap: wrap; }
  .controls-right { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }

  .btn {
    padding: 7px 14px; border: 1px solid var(--border); border-radius: var(--radius-sm);
    font-size: 12px; font-weight: 500; cursor: pointer;
    transition: all var(--transition); display: inline-flex; align-items: center; gap: 5px;
    background: var(--card); color: var(--text2);
  }
  .btn:hover { border-color: var(--border-hover); background: var(--bg2); color: var(--text); }
  .btn:active { transform: scale(.97); }
  .btn-primary { background: var(--accent); color: #fff; border-color: var(--accent); }
  .btn-primary:hover { background: #6a5ad9; border-color: #6a5ad9; }
  .btn-danger { background: rgba(248,113,113,.12); color: var(--red); border-color: rgba(248,113,113,.3); }
  .btn-danger:hover { background: rgba(248,113,113,.2); }
  .btn-success { background: rgba(45,212,191,.12); color: var(--green); border-color: rgba(45,212,191,.3); }
  .btn:disabled { opacity: .3; cursor: not-allowed; pointer-events: none; }
  .btn kbd {
    font-size: 9px; padding: 1px 5px; border-radius: 3px;
    background: rgba(255,255,255,.1); font-family: inherit;
    opacity: .6;
  }

  select, .form-select {
    padding: 7px 10px; border: 1px solid var(--border); border-radius: var(--radius-sm);
    background: var(--card); color: var(--text2); font-size: 12px;
    cursor: pointer; outline: none; transition: border-color var(--transition);
  }
  select:focus, .form-select:focus { border-color: var(--accent); }

  .theme-toggle {
    width: 32px; height: 32px; border-radius: var(--radius-sm); border: 1px solid var(--border);
    background: var(--card); cursor: pointer; font-size: 14px;
    display: flex; align-items: center; justify-content: center;
    transition: all var(--transition);
  }
  .theme-toggle:hover { border-color: var(--border-hover); }

  /* ── Main Layout ── */
  .main {
    display: grid;
    grid-template-columns: 1fr 1.5fr;
    gap: 12px;
    padding: 12px 20px;
    height: calc(100vh - 110px);
  }
  .panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    display: flex; flex-direction: column;
    overflow: hidden;
  }
  .panel-header {
    padding: 12px 16px;
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: .8px; color: var(--muted);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    display: flex; align-items: center; justify-content: space-between;
  }
  .panel-count {
    background: var(--accent-glow); color: var(--accent);
    padding: 2px 8px; border-radius: 10px; font-size: 10px; font-weight: 700;
    min-width: 24px; text-align: center;
  }
  .panel-body {
    padding: 12px 14px;
    overflow-y: auto; flex: 1;
    scroll-behavior: smooth;
  }
  .panel-body::-webkit-scrollbar { width: 4px; }
  .panel-body::-webkit-scrollbar-track { background: transparent; }
  .panel-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
  .panel-body::-webkit-scrollbar-thumb:hover { background: var(--border-hover); }

  /* ── Live Transcription ── */
  .live-text {
    font-size: calc(var(--font-size) + 1px); line-height: 1.8;
    color: var(--muted); font-style: italic;
    min-height: 60px;
  }
  .live-text.active {
    color: var(--text); font-style: normal;
    font-size: calc(var(--font-size) + 3px);
  }
  .cursor-blink::after {
    content: '▎'; animation: blink 1s step-end infinite; color: var(--accent);
    font-weight: 100;
  }
  @keyframes blink { 50% { opacity: 0; } }

  /* ── Q&A Items ── */
  .qa-item {
    margin-bottom: 10px; padding: 14px;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    animation: slideIn .35s var(--transition);
    transition: border-color .2s, box-shadow .2s;
  }
  .qa-item:hover {
    border-color: var(--border-hover);
    box-shadow: 0 2px 12px rgba(0,0,0,.15);
  }
  .qa-item.latest { border-color: var(--accent); border-left: 3px solid var(--accent); }
  @keyframes slideIn {
    from { opacity:0; transform:translateY(8px); }
    to { opacity:1; transform:none; }
  }
  .qa-section { margin-bottom: 10px; }
  .qa-section:last-child { margin-bottom: 0; }
  .qa-label {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .8px; margin-bottom: 4px;
    display: flex; align-items: center; gap: 5px;
  }
  .qa-label.original { color: var(--orange); }
  .qa-label.translated { color: var(--accent2); }
  .qa-label.answer-target { color: var(--green); }
  .qa-label.answer-input { color: var(--blue); }
  .qa-text { font-size: var(--font-size); line-height: 1.6; color: var(--text2); }
  .qa-answer-box {
    font-size: var(--font-size); line-height: 1.65;
    background: var(--green-bg);
    border-left: 2px solid var(--green);
    padding: 10px 14px; border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    white-space: pre-wrap; color: var(--text);
  }
  .qa-answer-box.en-box {
    border-left-color: var(--blue);
    background: rgba(96,165,250,.05);
  }
  .qa-timestamp {
    font-size: 10px; color: var(--muted); margin-top: 6px;
    text-align: right; opacity: .6;
  }
  .qa-copy-btn {
    float: right; background: none; border: 1px solid transparent;
    color: var(--muted); padding: 2px 6px; border-radius: 4px;
    font-size: 10px; cursor: pointer; transition: all var(--transition);
    opacity: .5;
  }
  .qa-copy-btn:hover { opacity: 1; color: var(--accent); border-color: var(--border); }

  /* ── Loading ── */
  .loading { display: inline-flex; gap: 5px; padding: 12px 0; }
  .loading span {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--accent); animation: bounce .6s infinite alternate;
  }
  .loading span:nth-child(2) { animation-delay: .15s; }
  .loading span:nth-child(3) { animation-delay: .3s; }
  @keyframes bounce { to { opacity:.2; transform:translateY(-8px); } }

  .hint {
    text-align: center; padding: 50px 20px;
    color: var(--muted); font-size: 13px; line-height: 1.8;
  }
  .hint-icon { font-size: 36px; margin-bottom: 10px; opacity: .7; }
  .hint-keys { margin-top: 14px; font-size: 11px; }
  .hint-keys kbd {
    padding: 2px 7px; border-radius: 4px;
    background: var(--bg2); border: 1px solid var(--border);
    font-family: inherit; font-size: 10px;
  }

  /* ── Toast ── */
  .toast {
    position: fixed; bottom: 20px; right: 20px;
    padding: 10px 18px; border-radius: var(--radius-sm);
    background: var(--card); border: 1px solid var(--border);
    box-shadow: var(--shadow-lg); font-size: 12px;
    transform: translateY(80px); opacity: 0;
    transition: all .3s cubic-bezier(.16,1,.3,1);
    z-index: 200; backdrop-filter: blur(12px);
  }
  .toast.show { transform: translateY(0); opacity: 1; }

  /* ── Settings Panel ── */
  .settings-overlay {
    position: fixed; inset: 0; background: rgba(0,0,0,.5);
    backdrop-filter: blur(4px);
    z-index: 300; display: none; align-items: center; justify-content: center;
  }
  .settings-overlay.open { display: flex; }
  .settings-panel {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 16px; padding: 0; width: 540px; max-width: 92vw;
    max-height: 85vh; overflow: hidden;
    box-shadow: var(--shadow-lg);
    display: flex; flex-direction: column;
  }
  .settings-header {
    padding: 20px 24px 0; flex-shrink: 0;
  }
  .settings-header h2 { font-size: 18px; margin-bottom: 16px; }
  .settings-tabs {
    display: flex; gap: 0; border-bottom: 1px solid var(--border);
  }
  .settings-tab {
    padding: 10px 20px; font-size: 13px; font-weight: 500;
    color: var(--muted); cursor: pointer; border: none; background: none;
    border-bottom: 2px solid transparent; transition: all .2s;
  }
  .settings-tab:hover { color: var(--text); }
  .settings-tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .settings-body {
    padding: 20px 24px; overflow-y: auto; flex: 1;
  }
  .settings-body::-webkit-scrollbar { width: 5px; }
  .settings-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
  .settings-group { margin-bottom: 16px; }
  .settings-group label {
    display: block; font-size: 11px; font-weight: 600;
    color: var(--muted); margin-bottom: 6px; text-transform: uppercase;
    letter-spacing: .5px;
  }
  .settings-group .hint-text {
    font-size: 11px; color: var(--muted); margin-top: 4px;
    font-style: italic;
  }
  .settings-group input, .settings-group select, .settings-group textarea {
    width: 100%; padding: 10px 14px; border: 1px solid var(--border);
    border-radius: 8px; background: var(--bg2); color: var(--text);
    font-size: 14px; font-family: inherit;
  }
  .settings-group textarea { resize: vertical; min-height: 70px; }
  .settings-group input:focus, .settings-group select:focus, .settings-group textarea:focus {
    border-color: var(--accent); outline: none;
  }
  .settings-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .settings-footer {
    padding: 16px 24px; border-top: 1px solid var(--border);
    display: flex; gap: 10px; justify-content: flex-end; flex-shrink: 0;
  }
  .settings-footer .btn { min-width: 100px; justify-content: center; }

  /* ── Connection Status ── */
  .conn-status {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600;
  }
  .conn-status.ok { background: rgba(0,206,201,.1); color: var(--green); }
  .conn-status.err { background: rgba(255,107,107,.1); color: var(--red); }

  /* ── Mic Visualizer ── */
  .mic-viz {
    display: flex; align-items: center; gap: 2px; height: 20px;
    padding: 0 8px; opacity: 0; transition: opacity .3s;
  }
  .mic-viz.active { opacity: 1; }
  .mic-viz-bar {
    width: 3px; border-radius: 2px;
    background: var(--green);
    transition: height .08s;
    min-height: 3px;
  }
  .mic-viz-bar:nth-child(1) { animation: vizBounce .4s ease infinite alternate; }
  .mic-viz-bar:nth-child(2) { animation: vizBounce .35s ease infinite alternate .05s; }
  .mic-viz-bar:nth-child(3) { animation: vizBounce .45s ease infinite alternate .1s; }
  .mic-viz-bar:nth-child(4) { animation: vizBounce .38s ease infinite alternate .15s; }
  .mic-viz-bar:nth-child(5) { animation: vizBounce .42s ease infinite alternate .08s; }
  @keyframes vizBounce {
    0% { height: 3px; opacity: .4; }
    100% { height: 18px; opacity: 1; }
  }

  /* ── Provider Indicator ── */
  .provider-wrap {
    display: flex; align-items: center; gap: 6px; position: relative;
  }
  .provider-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--red); transition: background .3s;
    flex-shrink: 0;
  }
  .provider-dot.ok { background: var(--green); }

  /* ── Processing Badge ── */
  .processing-badge {
    display: none; align-items: center; gap: 5px;
    padding: 3px 10px; border-radius: 20px;
    background: rgba(124,108,240,.12); color: var(--accent);
    font-size: 11px; font-weight: 600;
    animation: pulseBadge 1.5s infinite;
  }
  .processing-badge.active { display: inline-flex; }
  @keyframes pulseBadge {
    0%,100% { opacity: 1; }
    50% { opacity: .5; }
  }
  .processing-badge .spinner {
    width: 12px; height: 12px; border: 2px solid var(--accent);
    border-top-color: transparent; border-radius: 50%;
    animation: spin .6s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Collapsible Answers ── */
  .qa-collapse-btn {
    background: none; border: 1px solid var(--border);
    color: var(--muted); padding: 3px 10px; border-radius: 4px;
    font-size: 10px; cursor: pointer; transition: all var(--transition);
    margin-top: 6px;
  }
  .qa-collapse-btn:hover { color: var(--accent); border-color: var(--accent); }
  .qa-answers { transition: max-height .3s ease, opacity .3s; overflow: hidden; }
  .qa-answers.collapsed { max-height: 0 !important; opacity: 0; margin: 0; }

  /* ── Mode Switch ── */
  .mode-switch {
    display: flex; position: relative; border-radius: 8px;
    border: 1px solid var(--border); overflow: hidden;
    cursor: pointer; user-select: none;
    background: var(--bg2);
  }
  .mode-option {
    padding: 7px 16px; font-size: 12px; font-weight: 600;
    color: var(--muted); position: relative; z-index: 1;
    transition: color .3s; white-space: nowrap;
  }
  .mode-option.active { color: #fff; }
  .mode-slider {
    position: absolute; top: 2px; left: 2px; bottom: 2px;
    width: calc(50% - 2px); border-radius: 6px;
    background: var(--accent); transition: transform .3s cubic-bezier(.16,1,.3,1);
  }
  .mode-switch.conversation .mode-slider {
    transform: translateX(100%);
  }

  @media (max-width: 900px) {
    .main { grid-template-columns: 1fr; height: auto; }
    .panel { max-height: 50vh; }
    .controls { flex-direction: column; align-items: stretch; }
    .controls-left, .controls-right { justify-content: center; }
    .settings-row { grid-template-columns: 1fr; }
  }
</style>
</head>
<body data-theme="dark">

<!-- ── Header ── -->
<div class="header">
  <div class="brand">
    <h1>Interview Helper</h1>
    <span class="brand-badge">AI-Powered</span>
  </div>
  <div class="header-right">
    <div class="header-stats">
      <span id="timerDisplay" title="Session time">⏱ 00:00</span>
      <span title="Questions detected">💬 <span id="headerQaCount">0</span></span>
      <span class="processing-badge" id="processingBadge"><span class="spinner"></span> Processing</span>
    </div>
    <div class="status-bar">
      <div class="status-dot" id="statusDot"></div>
      <span id="statusText">Ready</span>
    </div>
    <button class="theme-toggle" id="themeToggle" title="Toggle theme (T)">🌙</button>
  </div>
</div>

<!-- ── Controls ── -->
<div class="controls">
  <div class="controls-left">
    <button class="btn btn-primary" id="btnStart" onclick="toggleListening()">
      ▶ Start <kbd>Space</kbd>
    </button>
    <button class="btn btn-danger" id="btnStop" onclick="stopListening()" disabled>
      ⏹ Stop <kbd>Space</kbd>
    </button>
    <button class="btn" onclick="clearHistory()" title="Clear history (Escape)">
      🗑 Clear <kbd>Esc</kbd>
    </button>
    <button class="btn" onclick="exportHistory()" title="Export (Ctrl+E)">
      📥 Export <kbd>Ctrl+E</kbd>
    </button>
    <button class="btn" onclick="openSettings()" title="Settings (Ctrl+,)">
      ⚙️ Settings <kbd>Ctrl+,</kbd>
    </button>
  </div>
  <div class="controls-right">
    <div class="mode-switch" id="modeSwitch" onclick="toggleMode()" title="Switch mode (M)">
      <div class="mode-option active" id="modeInterview">🎯 Interview</div>
      <div class="mode-option" id="modeConversation">💬 Conversation</div>
      <div class="mode-slider" id="modeSlider"></div>
    </div>
    <select id="inputLang" title="Input language" onchange="onInputLangChange()">
      %%LANG_OPTIONS_INPUT%%
    </select>
    <span style="color:var(--muted)">→</span>
    <select id="targetLang" title="Target language">
      %%LANG_OPTIONS_TARGET%%
    </select>
    <div class="provider-wrap">
      <span class="provider-dot" id="providerDot"></span>
      <select id="providerSelect" title="AI Provider" onchange="updateProviderDot()">
        %%PROVIDER_OPTIONS%%
      </select>
    </div>
  </div>
</div>

<!-- ── Main ── -->
<div class="main">
  <div class="panel">
    <div class="panel-header">
      <span>📝 Live Transcription</span>
      <div class="mic-viz" id="micViz">
        <div class="mic-viz-bar"></div>
        <div class="mic-viz-bar"></div>
        <div class="mic-viz-bar"></div>
        <div class="mic-viz-bar"></div>
        <div class="mic-viz-bar"></div>
      </div>
    </div>
    <div class="panel-body">
      <div class="live-text cursor-blink" id="liveText">
        Press <kbd>Space</kbd> or click <strong>Start</strong> to begin listening...
      </div>
    </div>
  </div>

  <div class="panel">
    <div class="panel-header">
      <span>💡 Questions & Answers</span>
      <span class="panel-count" id="qaCount">0</span>
    </div>
    <div class="panel-body" id="qaPanel">
      <div class="hint" id="qaHint">
        <div class="hint-icon">🎯</div>
        Detected questions and AI-generated answer suggestions appear here
        <div class="hint-keys">
          <kbd>Space</kbd> Start/Stop &nbsp;
          <kbd>Esc</kbd> Clear &nbsp;
          <kbd>Ctrl+E</kbd> Export &nbsp;
          <kbd>T</kbd> Theme &nbsp;
          <kbd>M</kbd> Mode
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ── Settings Modal ── -->
<div class="settings-overlay" id="settingsOverlay" onclick="if(event.target===this)closeSettings()">
  <div class="settings-panel">
    <div class="settings-header">
      <h2>⚙️ Settings</h2>
      <div class="settings-tabs">
        <button class="settings-tab active" onclick="switchTab('ai')">🤖 AI Provider</button>
        <button class="settings-tab" onclick="switchTab('interview')">🎯 Interview</button>
        <button class="settings-tab" onclick="switchTab('display')">🎨 Display</button>
      </div>
    </div>

    <div class="settings-body">
      <!-- Tab: AI Provider -->
      <div class="tab-content active" id="tab-ai">
        <div class="settings-group">
          <label>AI Provider</label>
          <select id="settingsProvider" onchange="updateSettingsFields()">
            %%PROVIDER_OPTIONS%%
          </select>
        </div>
        <div class="settings-group">
          <label>API Key</label>
          <input type="password" id="settingsApiKey" placeholder="Enter your API key...">
          <div class="hint-text">Leave empty to clear. Ollama needs no key.</div>
        </div>
        <div class="settings-row">
          <div class="settings-group">
            <label>Base URL</label>
            <input type="text" id="settingsBaseUrl" placeholder="https://...">
          </div>
          <div class="settings-group">
            <label>Model</label>
            <input type="text" id="settingsModel" placeholder="Model name...">
          </div>
        </div>
        <div class="settings-group">
          <label>Connection Test</label>
          <button class="btn" onclick="testConnection()" id="btnTestConn">🔌 Test Connection</button>
          <span id="connResult" style="margin-left:10px;"></span>
        </div>
      </div>

      <!-- Tab: Interview -->
      <div class="tab-content" id="tab-interview">
        <div class="settings-group">
          <label>Interview Context</label>
          <textarea id="settingsContext" placeholder="e.g. Applying for Senior Developer at Google. 5 years experience in Python and React. Strengths: problem-solving, team leadership."></textarea>
          <div class="hint-text">Help the AI generate more personalized and relevant answers.</div>
        </div>
        <div class="settings-row">
          <div class="settings-group">
            <label>Answer Length</label>
            <select id="settingsLength">
              <option value="short">Short (1-2 sentences)</option>
              <option value="medium" selected>Medium (3-5 sentences)</option>
              <option value="long">Long (5-8 sentences)</option>
            </select>
          </div>
          <div class="settings-group">
            <label>Answer Tone</label>
            <select id="settingsTone">
              <option value="professional" selected>Professional</option>
              <option value="casual">Casual / Friendly</option>
              <option value="technical">Technical / Detailed</option>
              <option value="confident">Confident / Assertive</option>
              <option value="humble">Humble / Reflective</option>
            </select>
          </div>
        </div>
        <div class="settings-group">
          <label>Custom Instructions</label>
          <textarea id="settingsCustomPrompt" placeholder="e.g. Always mention my experience with distributed systems. Focus on leadership examples. Avoid talking about salary."></textarea>
          <div class="hint-text">Additional instructions for the AI when generating answers.</div>
        </div>
        <div class="settings-group">
          <label>Silence Delay (ms)</label>
          <input type="range" id="settingsSilence" min="500" max="6000" step="250" value="2000"
                 oninput="document.getElementById('silenceVal').textContent=this.value+'ms'">
          <span id="silenceVal" style="font-size:13px;color:var(--muted);">2000ms</span>
          <div class="hint-text">How long to wait after silence before processing. Lower = faster but may cut sentences.</div>
        </div>
      </div>

      <!-- Tab: Display -->
      <div class="tab-content" id="tab-display">
        <div class="settings-row">
          <div class="settings-group">
            <label>Font Size</label>
            <select id="settingsFontSize">
              <option value="12">Small (12px)</option>
              <option value="14" selected>Medium (14px)</option>
              <option value="16">Large (16px)</option>
              <option value="18">Extra Large (18px)</option>
            </select>
          </div>
          <div class="settings-group">
            <label>Theme</label>
            <select id="settingsTheme" onchange="applyTheme(this.value)">
              <option value="dark">Dark</option>
              <option value="light">Light</option>
            </select>
          </div>
        </div>
        <div class="settings-row">
          <div class="settings-group">
            <label>Sound Effects</label>
            <select id="settingsSound">
              <option value="on">On</option>
              <option value="off" selected>Off</option>
            </select>
          </div>
          <div class="settings-group">
            <label>Auto-Scroll</label>
            <select id="settingsAutoScroll">
              <option value="on" selected>On</option>
              <option value="off">Off</option>
            </select>
          </div>
        </div>
        <div class="settings-group">
          <label>Show Sections</label>
          <div style="display:flex;gap:14px;margin-top:6px;flex-wrap:wrap;">
            <label style="display:flex;align-items:center;gap:6px;font-size:13px;text-transform:none;letter-spacing:0;color:var(--text);cursor:pointer;">
              <input type="checkbox" id="showOriginal" checked> Original
            </label>
            <label style="display:flex;align-items:center;gap:6px;font-size:13px;text-transform:none;letter-spacing:0;color:var(--text);cursor:pointer;">
              <input type="checkbox" id="showTranslation" checked> Translation
            </label>
            <label style="display:flex;align-items:center;gap:6px;font-size:13px;text-transform:none;letter-spacing:0;color:var(--text);cursor:pointer;">
              <input type="checkbox" id="showAnswerTarget" checked> Answer (Target)
            </label>
            <label style="display:flex;align-items:center;gap:6px;font-size:13px;text-transform:none;letter-spacing:0;color:var(--text);cursor:pointer;">
              <input type="checkbox" id="showAnswerInput" checked> Answer (Input)
            </label>
          </div>
        </div>
      </div>
    </div>

    <div class="settings-footer">
      <button class="btn" onclick="closeSettings()">Cancel</button>
      <button class="btn btn-primary" onclick="saveSettings()">💾 Save Settings</button>
    </div>
  </div>
</div>

<!-- ── Toast ── -->
<div class="toast" id="toast"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.4/socket.io.min.js"></script>
<script>
const socket = io();
let recognition = null;
let isListening = false;
let finalTranscript = '';
let interimTranscript = '';
let silenceTimer = null;
let SILENCE_DELAY = 2000;
let qaItems = [];
let soundEnabled = false;
let autoScrollEnabled = true;
let currentMode = 'interview'; // 'interview' or 'conversation'

// ── Provider Status Dot ───────────────────────────────────────────────
let processingCount = 0;
function updateProviderDot() {
  const p = document.getElementById('providerSelect').value;
  socket.emit('check_provider', { provider: p }, (res) => {
    const dot = document.getElementById('providerDot');
    dot.className = res && res.ok ? 'provider-dot ok' : 'provider-dot';
  });
}
// Check on load
setTimeout(updateProviderDot, 500);

function setProcessing(active) {
  processingCount += active ? 1 : -1;
  processingCount = Math.max(0, processingCount);
  const badge = document.getElementById('processingBadge');
  badge.className = processingCount > 0 ? 'processing-badge active' : 'processing-badge';
}

function toggleCollapse(id) {
  const el = document.getElementById('answers-' + id);
  const btn = document.getElementById('colbtn-' + id);
  if (!el) return;
  if (el.classList.contains('collapsed')) {
    el.classList.remove('collapsed');
    el.style.maxHeight = el.scrollHeight + 'px';
    btn.textContent = '▾ Collapse';
  } else {
    el.classList.add('collapsed');
    btn.textContent = '▸ Show answers';
  }
}

// ── Session Timer ─────────────────────────────────────────────────────
let timerInterval = null;
let timerSeconds = 0;
function startTimer() {
  if (timerInterval) return;
  timerInterval = setInterval(() => {
    timerSeconds++;
    const m = String(Math.floor(timerSeconds / 60)).padStart(2, '0');
    const s = String(timerSeconds % 60).padStart(2, '0');
    document.getElementById('timerDisplay').textContent = '⏱ ' + m + ':' + s;
  }, 1000);
}
function stopTimer() {
  clearInterval(timerInterval);
  timerInterval = null;
}
function resetTimer() {
  stopTimer();
  timerSeconds = 0;
  document.getElementById('timerDisplay').textContent = '⏱ 00:00';
}

// ── Reusable AudioContext (fix: no leak) ──────────────────────────────
let _audioCtx = null;
function getAudioCtx() {
  if (!_audioCtx || _audioCtx.state === 'closed') {
    _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (_audioCtx.state === 'suspended') _audioCtx.resume();
  return _audioCtx;
}

function playTick() {
  if (!soundEnabled) return;
  try {
    const ctx = getAudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain); gain.connect(ctx.destination);
    osc.frequency.value = 880;
    gain.gain.setValueAtTime(0.08, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.1);
    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + 0.1);
  } catch(e) {}
}

function playDone() {
  if (!soundEnabled) return;
  try {
    const ctx = getAudioCtx();
    [660, 880].forEach((f, i) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain); gain.connect(ctx.destination);
      osc.frequency.value = f;
      gain.gain.setValueAtTime(0.06, ctx.currentTime + i*0.12);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + i*0.12 + 0.15);
      osc.start(ctx.currentTime + i*0.12);
      osc.stop(ctx.currentTime + i*0.12 + 0.15);
    });
  } catch(e) {}
}

// ── Theme ─────────────────────────────────────────────────────────────
function applyTheme(theme) {
  document.body.setAttribute('data-theme', theme);
  document.getElementById('themeToggle').textContent = theme === 'dark' ? '🌙' : '☀️';
  document.getElementById('settingsTheme').value = theme;
}
function toggleTheme() {
  const isDark = document.body.getAttribute('data-theme') === 'dark';
  applyTheme(isDark ? 'light' : 'dark');
}
document.getElementById('themeToggle').onclick = toggleTheme;

// ── Settings Tabs ─────────────────────────────────────────────────────
function switchTab(tab) {
  document.querySelectorAll('.settings-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('tab-' + tab).classList.add('active');
}

// ── Mode Switch ──────────────────────────────────────────────────────
function toggleMode() {
  currentMode = currentMode === 'interview' ? 'conversation' : 'interview';
  const sw = document.getElementById('modeSwitch');
  const mI = document.getElementById('modeInterview');
  const mC = document.getElementById('modeConversation');
  if (currentMode === 'conversation') {
    sw.classList.add('conversation');
    mI.classList.remove('active');
    mC.classList.add('active');
    showToast('💬 Conversation mode — translation only');
  } else {
    sw.classList.remove('conversation');
    mC.classList.remove('active');
    mI.classList.add('active');
    showToast('🎯 Interview mode — translation + answers');
  }
}

// ── Speech Recognition ────────────────────────────────────────────────
function initRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    showToast('❌ Browser does not support Speech Recognition. Use Chrome or Edge.');
    return null;
  }
  const r = new SR();
  r.lang = document.getElementById('inputLang').value;
  r.continuous = true;
  r.interimResults = true;
  r.maxAlternatives = 1;

  r.onresult = (e) => {
    interimTranscript = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const t = e.results[i][0].transcript;
      if (e.results[i].isFinal) {
        finalTranscript += t + ' ';
        clearTimeout(silenceTimer);
        silenceTimer = setTimeout(() => processSentence(), SILENCE_DELAY);
      } else {
        interimTranscript += t;
      }
    }
    updateLiveText();
  };

  r.onerror = (e) => {
    if (e.error === 'no-speech') return;
    if (e.error === 'aborted') return;
    if (e.error === 'not-allowed') {
      showToast('🚫 Microphone access denied. Allow it in browser settings.');
      stopListening();
    } else if (e.error === 'network') {
      showToast('🌐 Network error — check internet connection.');
    } else {
      showToast('⚠️ Speech error: ' + e.error);
    }
  };

  r.onend = () => {
    if (isListening) {
      try { r.start(); } catch(err) {}
    }
  };

  return r;
}

function onInputLangChange() {
  // Restart recognition with new language if currently listening
  if (isListening && recognition) {
    try { recognition.stop(); } catch(e) {}
    setTimeout(() => {
      recognition = initRecognition();
      if (recognition) recognition.start();
    }, 200);
  }
}

function updateLiveText() {
  const el = document.getElementById('liveText');
  const full = finalTranscript + interimTranscript;
  if (full) {
    el.textContent = full;
    el.className = 'live-text active cursor-blink';
  } else {
    el.textContent = 'Waiting for speech...';
    el.className = 'live-text cursor-blink';
  }
}

function processSentence() {
  const text = finalTranscript.trim();
  if (!text || text.length < 8) return;

  const inputLang = document.getElementById('inputLang').value;
  const targetLang = document.getElementById('targetLang').value;
  const provider = document.getElementById('providerSelect').value;

  socket.emit('process_text', {
    text, input_lang: inputLang, target_lang: targetLang, provider,
    mode: currentMode
  });
  playTick();
  finalTranscript = '';
  interimTranscript = '';
  updateLiveText();
}

// ── Controls ──────────────────────────────────────────────────────────
function toggleListening() { isListening ? stopListening() : startListening(); }

function startListening() {
  if (recognition) { try { recognition.stop(); } catch(e){} }
  recognition = initRecognition();
  if (!recognition) return;

  finalTranscript = '';
  interimTranscript = '';
  isListening = true;
  recognition.start();

  document.getElementById('btnStart').disabled = true;
  document.getElementById('btnStop').disabled = false;
  document.getElementById('statusDot').className = 'status-dot listening';
  document.getElementById('statusText').textContent = 'Listening...';
  document.getElementById('liveText').textContent = 'Waiting for speech...';
  document.getElementById('liveText').className = 'live-text cursor-blink';
  document.getElementById('micViz').classList.add('active');
  startTimer();
  showToast('🎤 Listening started');
}

function stopListening() {
  isListening = false;
  if (recognition) { try { recognition.stop(); } catch(e){} }
  clearTimeout(silenceTimer);
  if (finalTranscript.trim().length > 8) processSentence();

  document.getElementById('btnStart').disabled = false;
  document.getElementById('btnStop').disabled = true;
  document.getElementById('statusDot').className = 'status-dot';
  document.getElementById('statusText').textContent = 'Stopped';
  document.getElementById('micViz').classList.remove('active');
  stopTimer();
  showToast('⏹ Listening stopped');
}

function clearHistory() {
  qaItems = [];
  document.getElementById('qaPanel').innerHTML = `
    <div class="hint" id="qaHint">
      <div class="hint-icon">🎯</div>
      Detected questions and AI-generated answer suggestions appear here
      <div class="hint-keys">
        <kbd>Space</kbd> Start/Stop &nbsp;
        <kbd>Esc</kbd> Clear &nbsp;
        <kbd>Ctrl+E</kbd> Export &nbsp;
        <kbd>T</kbd> Theme
      </div>
    </div>`;
  document.getElementById('qaCount').textContent = '0';
  document.getElementById('headerQaCount').textContent = '0';
  resetTimer();
  socket.emit('clear_history');
  showToast('🗑 History cleared');
}

// ── Copy to Clipboard ─────────────────────────────────────────────────
function copyAnswer(text) {
  navigator.clipboard.writeText(text).then(() => showToast('📋 Copied!'));
}

// ── Export ─────────────────────────────────────────────────────────────
function exportHistory() {
  if (!qaItems.length) { showToast('⚠️ Nothing to export'); return; }

  let md = '# Interview Helper — Transcript\n\n';
  md += `**Date:** ${new Date().toLocaleDateString()}\n`;
  md += `**Questions:** ${qaItems.length}\n\n---\n\n`;

  qaItems.forEach((item, i) => {
    md += `## Q${i+1}\n\n`;
    md += `**Original:** ${item.original}\n\n`;
    if (item.translation) md += `**Translation:** ${item.translation}\n\n`;
    md += `**Answer (${item.target_lang}):** ${item.answer_target}\n\n`;
    md += `**Answer (${item.input_lang}):** ${item.answer_input}\n\n`;
    md += `---\n\n`;
  });

  const blob = new Blob([md], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `interview_${new Date().toISOString().slice(0,10)}.md`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('📥 Exported as Markdown');
}

// ── Settings ──────────────────────────────────────────────────────────
function openSettings() {
  document.getElementById('settingsOverlay').classList.add('open');
  document.getElementById('settingsTheme').value = document.body.getAttribute('data-theme');
  updateSettingsFields();
}
function closeSettings() {
  document.getElementById('settingsOverlay').classList.remove('open');
}
function updateSettingsFields() {
  const p = document.getElementById('settingsProvider').value;
  socket.emit('get_provider_config', { provider: p }, (cfg) => {
    if (!cfg) return;
    document.getElementById('settingsApiKey').value = cfg.api_key_set ? '' : '';
    document.getElementById('settingsApiKey').placeholder = cfg.api_key_set ? '••••••• (key is set, leave empty to keep)' : 'Enter your API key...';
    document.getElementById('settingsBaseUrl').value = cfg.base_url || '';
    document.getElementById('settingsModel').value = cfg.model || '';
    document.getElementById('settingsContext').value = cfg.context || '';
    document.getElementById('settingsLength').value = cfg.answer_length || 'medium';
    document.getElementById('settingsTone').value = cfg.answer_tone || 'professional';
    document.getElementById('settingsCustomPrompt').value = cfg.custom_prompt || '';
  });
  document.getElementById('connResult').innerHTML = '';
}

function saveSettings() {
  const provider = document.getElementById('settingsProvider').value;
  const apiKey = document.getElementById('settingsApiKey').value;
  const baseUrl = document.getElementById('settingsBaseUrl').value;
  const model = document.getElementById('settingsModel').value;
  const silence = parseInt(document.getElementById('settingsSilence').value) || 2000;
  const fontSize = document.getElementById('settingsFontSize').value;
  const sound = document.getElementById('settingsSound').value;
  const autoScroll = document.getElementById('settingsAutoScroll').value;
  const context = document.getElementById('settingsContext').value;
  const answerLength = document.getElementById('settingsLength').value;
  const answerTone = document.getElementById('settingsTone').value;
  const customPrompt = document.getElementById('settingsCustomPrompt').value;

  SILENCE_DELAY = silence;
  soundEnabled = sound === 'on';
  autoScrollEnabled = autoScroll === 'on';
  document.documentElement.style.setProperty('--font-size', fontSize + 'px');

  socket.emit('update_settings', {
    provider,
    api_key: apiKey || null,
    base_url: baseUrl,
    model,
    context, answer_length: answerLength,
    answer_tone: answerTone, custom_prompt: customPrompt,
  });

  document.getElementById('providerSelect').value = provider;
  updateProviderDot();
  closeSettings();
  showToast('✅ Settings saved');
}

function testConnection() {
  const provider = document.getElementById('settingsProvider').value;
  const btn = document.getElementById('btnTestConn');
  const result = document.getElementById('connResult');
  btn.disabled = true;
  result.innerHTML = '<span style="color:var(--muted)">Testing...</span>';

  socket.emit('test_connection', { provider }, (res) => {
    btn.disabled = false;
    if (res && res.ok) {
      result.innerHTML = '<span class="conn-status ok">✅ Connected</span>';
    } else {
      result.innerHTML = `<span class="conn-status err">❌ ${escHtml(res?.error || 'Failed')}</span>`;
    }
  });
}

// ── Keyboard Shortcuts ────────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
  if (['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName)) return;

  if (e.code === 'Space') {
    e.preventDefault();
    toggleListening();
  } else if (e.code === 'Escape') {
    if (document.getElementById('settingsOverlay').classList.contains('open')) {
      closeSettings();
    } else {
      clearHistory();
    }
  } else if (e.code === 'KeyE' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    exportHistory();
  } else if (e.code === 'Comma' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    openSettings();
  } else if (e.code === 'KeyT') {
    toggleTheme();
  } else if (e.code === 'KeyM') {
    toggleMode();
  }
});

// ── Socket Events ─────────────────────────────────────────────────────
socket.on('processing', (data) => {
  const hint = document.getElementById('qaHint');
  if (hint) hint.remove();

  const div = document.createElement('div');
  div.className = 'qa-item';
  div.id = 'qa-' + data.id;
  div.innerHTML = `
    <div class="qa-section">
      <div class="qa-label original">🗣 Original</div>
      <div class="qa-text">${escHtml(data.original)}</div>
    </div>
    <div class="loading"><span></span><span></span><span></span></div>
  `;
  const panel = document.getElementById('qaPanel');
  panel.prepend(div);
  setProcessing(true);
});

socket.on('result', (data) => {
  setProcessing(false);
  const div = document.getElementById('qa-' + data.id);
  if (!div) return;

  const targetName = data.target_lang_name || data.target_lang;
  const inputName = data.input_lang_name || data.input_lang;

  const showOrig = document.getElementById('showOriginal').checked;
  const showTrans = document.getElementById('showTranslation').checked;
  const showAT = document.getElementById('showAnswerTarget').checked;
  const showAI = document.getElementById('showAnswerInput').checked;
  const isConversation = data.mode === 'conversation';

  let html = '';
  if (showOrig) html += `
    <div class="qa-section">
      <div class="qa-label original">🗣 Original (${escHtml(inputName)})</div>
      <div class="qa-text">${escHtml(data.original)}</div>
    </div>`;
  if (showTrans) html += `
    <div class="qa-section">
      <div class="qa-label translated">🔄 Translation (${escHtml(targetName)})
        <button class="qa-copy-btn" onclick="copyAnswer('${escJs(data.translation)}')">📋 Copy</button>
      </div>
      <div class="qa-text">${escHtml(data.translation)}</div>
    </div>`;
  const hasAnswers = !isConversation && (
    (showAT && data.answer_target && data.answer_target !== '—') ||
    (showAI && data.answer_input && data.answer_input !== '—')
  );

  if (hasAnswers) {
    html += `<div class="qa-answers" id="answers-${data.id}" style="max-height:600px;">`;
    if (showAT && data.answer_target && data.answer_target !== '—') html += `
      <div class="qa-section">
        <div class="qa-label answer-target">💡 Answer (${escHtml(targetName)})
          <button class="qa-copy-btn" onclick="copyAnswer('${escJs(data.answer_target)}')">📋</button>
        </div>
        <div class="qa-answer-box">${escHtml(data.answer_target)}</div>
      </div>`;
    if (showAI && data.answer_input && data.answer_input !== '—') html += `
      <div class="qa-section">
        <div class="qa-label answer-input">💬 Answer (${escHtml(inputName)})
          <button class="qa-copy-btn" onclick="copyAnswer('${escJs(data.answer_input)}')">📋</button>
        </div>
        <div class="qa-answer-box en-box">${escHtml(data.answer_input)}</div>
      </div>`;
    html += `</div>`;
    html += `<button class="qa-collapse-btn" id="colbtn-${data.id}" onclick="toggleCollapse(${data.id})">▾ Collapse</button>`;
  }

  html += `<div class="qa-timestamp">${new Date().toLocaleTimeString()}</div>`;
  div.innerHTML = html;

  qaItems.push({
    original: data.original,
    translation: data.translation,
    answer_target: data.answer_target,
    answer_input: data.answer_input,
    target_lang: targetName,
    input_lang: inputName,
  });
  document.getElementById('qaCount').textContent = qaItems.length;
  document.getElementById('headerQaCount').textContent = qaItems.length;
  // Highlight latest item
  document.querySelectorAll('.qa-item.latest').forEach(el => el.classList.remove('latest'));
  div.classList.add('latest');
  playDone();

  if (autoScrollEnabled) {
    div.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
});

socket.on('error', (data) => {
  setProcessing(false);
  const div = document.getElementById('qa-' + data.id);
  if (div) {
    const loading = div.querySelector('.loading');
    if (loading) loading.remove();
    div.innerHTML += `<div style="color:var(--red);padding:8px 0;font-size:13px;">⚠️ ${escHtml(data.message)}</div>`;
  }
});

// ── Helpers ───────────────────────────────────────────────────────────
function escHtml(s) {
  if (!s) return '';
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function escJs(s) {
  if (!s) return '';
  return s.replace(/\\/g,'\\\\').replace(/'/g,"\\'").replace(/\n/g,'\\n').replace(/\r/g,'');
}

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove('show'), 2500);
}
</script>
</body>
</html>
"""

# ── Inject dynamic options into template ───────────────────────────────
def build_template():
    tpl = HTML_TEMPLATE

    lang_input = "\n".join(
        f'<option value="{SPEECH_LANG_CODES[k]}" {"selected" if k=="en" else ""}>'
        f'{v} ({k})</option>'
        for k, v in LANGUAGES.items()
    )
    lang_target = "\n".join(
        f'<option value="{k}" {"selected" if k=="de" else ""}>'
        f'{v} ({k})</option>'
        for k, v in LANGUAGES.items()
    )
    provider_opts = "\n".join(
        f'<option value="{k}" {"selected" if k=="qwen" else ""}>{v["name"]}</option>'
        for k, v in PROVIDERS.items()
    )

    tpl = tpl.replace("%%LANG_OPTIONS_INPUT%%", lang_input)
    tpl = tpl.replace("%%LANG_OPTIONS_TARGET%%", lang_target)
    tpl = tpl.replace("%%PROVIDER_OPTIONS%%", provider_opts)
    return tpl


RENDERED_TEMPLATE = None


# ── Routes ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    global RENDERED_TEMPLATE
    if not RENDERED_TEMPLATE:
        RENDERED_TEMPLATE = build_template()
    return render_template_string(RENDERED_TEMPLATE)


# ── Socket Handlers ────────────────────────────────────────────────────
@socketio.on("process_text")
def handle_process_text(data):
    global msg_counter
    text = data.get("text", "").strip()
    input_lang_code = data.get("input_lang", "en-US")
    target_lang = data.get("target_lang", "de")
    provider = data.get("provider", "qwen")
    mode = data.get("mode", "interview")  # 'interview' or 'conversation'

    if not text:
        return

    msg_counter += 1
    msg_id = msg_counter

    input_lang_key = input_lang_code.split("-")[0]
    input_lang_name = LANGUAGES.get(input_lang_key, input_lang_key)
    target_lang_name = LANGUAGES.get(target_lang, target_lang)

    emit("processing", {"id": msg_id, "original": text})

    cfg = PROVIDERS.get(provider, PROVIDERS["qwen"])
    if not cfg["api_key"]:
        emit("error", {
            "id": msg_id,
            "message": f"No API key for {cfg['name']}. Set via env var or ⚙️ Settings."
        })
        return

    conversation_history.append({"role": "user_question", "text": text})

    def process():
        try:
            # Build context from history
            context_str = ""
            if len(conversation_history) > 1:
                recent = conversation_history[-6:]
                context_str = "\n".join(f"- {item['text']}" for item in recent[:-1])
                context_str = f"\nPrevious questions in this interview:\n{context_str}\n"

            # Build answer length instruction
            length_map = {
                "short": "Keep each answer to 1-2 sentences.",
                "medium": "Keep each answer to 3-5 sentences.",
                "long": "Provide detailed answers of 5-8 sentences with examples.",
            }
            length_instr = length_map.get(
                interview_settings["answer_length"], length_map["medium"]
            )

            # Build tone instruction
            tone_map = {
                "professional": "Use a professional, polished tone.",
                "casual": "Use a casual, friendly, and conversational tone.",
                "technical": "Use a technical, detailed tone with specific terminology.",
                "confident": "Use a confident, assertive tone that shows conviction.",
                "humble": "Use a humble, reflective tone showing self-awareness.",
            }
            tone_instr = tone_map.get(
                interview_settings["answer_tone"], tone_map["professional"]
            )

            # Build interview context
            interview_ctx = ""
            if interview_settings["context"]:
                interview_ctx = (
                    f"\nAbout the interviewee: {interview_settings['context']}\n"
                    "Use this background to personalize answers with relevant "
                    "experience and examples.\n"
                )

            # Build custom prompt
            custom_instr = ""
            if interview_settings["custom_prompt"]:
                custom_instr = (
                    f"\nAdditional instructions: "
                    f"{interview_settings['custom_prompt']}\n"
                )

            if mode == "conversation":
                # Conversation mode: translation only, no answer suggestions
                prompt = f"""You are a real-time translation assistant.

Someone just said the following in {input_lang_name}:
"{text}"

Translate this text into {target_lang_name}. Preserve the tone and nuance.

Respond EXACTLY in this JSON format (no markdown, no codeblock, no extra text):
{{"translation": "...", "answer_target": "", "answer_input": ""}}"""
            else:
                # Interview mode: translation + answer suggestions
                prompt = f"""You are an interview assistant helping with a job interview.
{interview_ctx}
The interviewer just said the following in {input_lang_name}:
"{text}"
{context_str}
Please:
1. Translate the text into {target_lang_name}.
2. If it's a question or expects a response, provide a professional answer suggestion in {target_lang_name} AND in {input_lang_name}. {length_instr} {tone_instr}
3. If it's not a question (e.g. small talk, introduction), still translate and give a brief appropriate reaction in both languages.
{custom_instr}
Respond EXACTLY in this JSON format (no markdown, no codeblock, no extra text):
{{"translation": "...", "answer_target": "...", "answer_input": "..."}}"""

            result_text = call_ai(prompt, provider)

            # Parse JSON response
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", result_text, re.DOTALL)
                if match:
                    result = json.loads(match.group())
                else:
                    result = {
                        "translation": "⚠️ Could not parse AI response",
                        "answer_target": result_text,
                        "answer_input": result_text,
                    }

            socketio.emit("result", {
                "id": msg_id,
                "original": text,
                "translation": result.get("translation", "—"),
                "answer_target": result.get("answer_target", "—"),
                "answer_input": result.get("answer_input", "—"),
                "input_lang": input_lang_key,
                "target_lang": target_lang,
                "input_lang_name": input_lang_name,
                "target_lang_name": target_lang_name,
                "mode": mode,
            })

        except Exception as e:
            err_msg = str(e)
            # Make common errors more readable
            if "api_key" in err_msg.lower() or "auth" in err_msg.lower():
                err_msg = f"Authentication error — check your API key for {cfg['name']}."
            elif "model" in err_msg.lower() and "not" in err_msg.lower():
                err_msg = f"Model '{cfg['model']}' not found — check model name in Settings."
            elif "connection" in err_msg.lower() or "connect" in err_msg.lower():
                err_msg = f"Connection failed — check your internet or base URL."
            socketio.emit("error", {"id": msg_id, "message": err_msg})

    thread = threading.Thread(target=process, daemon=True)
    thread.start()


@socketio.on("clear_history")
def handle_clear():
    global conversation_history
    conversation_history = []


@socketio.on("get_provider_config")
def handle_get_config(data):
    provider = data.get("provider", "qwen")
    cfg = PROVIDERS.get(provider, {})
    return {
        "api_key_set": bool(cfg.get("api_key")),
        "base_url": cfg.get("base_url") or "",
        "model": cfg.get("model") or "",
        "context": interview_settings.get("context", ""),
        "answer_length": interview_settings.get("answer_length", "medium"),
        "answer_tone": interview_settings.get("answer_tone", "professional"),
        "custom_prompt": interview_settings.get("custom_prompt", ""),
    }


@socketio.on("update_settings")
def handle_update_settings(data):
    provider = data.get("provider", "qwen")
    if provider in PROVIDERS:
        api_key = data.get("api_key")
        if api_key is not None and api_key != "":
            PROVIDERS[provider]["api_key"] = api_key
        # Allow clearing key by sending empty string explicitly
        if api_key == "":
            pass  # keep existing key (empty field = no change)

        base_url = data.get("base_url")
        if base_url:
            PROVIDERS[provider]["base_url"] = base_url

        model = data.get("model")
        if model:
            PROVIDERS[provider]["model"] = model

    # Update interview settings
    if data.get("context") is not None:
        interview_settings["context"] = data["context"]
    if data.get("answer_length"):
        interview_settings["answer_length"] = data["answer_length"]
    if data.get("answer_tone"):
        interview_settings["answer_tone"] = data["answer_tone"]
    if data.get("custom_prompt") is not None:
        interview_settings["custom_prompt"] = data["custom_prompt"]

    # Invalidate cached template if provider changed
    global RENDERED_TEMPLATE
    RENDERED_TEMPLATE = None


@socketio.on("check_provider")
def handle_check_provider(data):
    provider = data.get("provider", "qwen")
    cfg = PROVIDERS.get(provider, {})
    return {"ok": bool(cfg.get("api_key"))}


@socketio.on("test_connection")
def handle_test_connection(data):
    provider = data.get("provider", "qwen")
    cfg = PROVIDERS.get(provider)
    if not cfg:
        return {"ok": False, "error": "Unknown provider"}
    if not cfg["api_key"]:
        return {"ok": False, "error": "No API key set"}

    try:
        result = call_ai("Say 'OK' in one word.", provider)
        return {"ok": True, "response": result[:50]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:120]}


# ── Start ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print()
    print("  ╔══════════════════════════════════════════╗")
    print("  ║     🎤 Interview Helper                  ║")
    print("  ║     Real-time AI Interview Assistant      ║")
    print("  ╠══════════════════════════════════════════╣")
    print(f"  ║  🌐 http://localhost:{DEFAULT_PORT:<21}  ║")
    print("  ║  📖 Press Space to start listening       ║")
    print("  ║  ⚙️  Ctrl+, for settings                 ║")
    print("  ╚══════════════════════════════════════════╝")
    print()

    configured = [k for k, v in PROVIDERS.items() if v["api_key"]]
    if configured:
        print(f"  ✅ Configured: {', '.join(configured)}")
    else:
        print("  ⚠️  No API keys set!")
        print("     Set via env var or use ⚙️ Settings in the UI")
    print()

    socketio.run(app, host="0.0.0.0", port=DEFAULT_PORT, debug=False)
