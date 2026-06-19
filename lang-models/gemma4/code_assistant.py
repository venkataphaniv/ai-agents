"""
Gemma 4 · Code Assistant
─────────────────────────────────────────────────────────────────────
Split-pane: Code Editor (left)  +  Agentic Chat (right)
Multimodal · Multilingual · Thinking mode · 100% local via Ollama
Gradio 6.0 compatible · Light/dark contrast themes
─────────────────────────────────────────────────────────────────────
Usage:
    pip install gradio requests pillow
    ollama pull gemma4:e4b
    python app.py  →  http://localhost:7860
"""


import base64 as b64
import json
import math
import os
import re
import subprocess as sp
import tempfile
from pathlib import Path

import gradio as gr
import requests as r


# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
OLLAMA_BASE = f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:11434"
MODEL = 'gemma4' # "gemma4:e4b"
DEBUG = True

LANGUAGES = [
    "python", "javascript", "typescript", "bash", "sql",
    "rust", "go", "java", "c", "cpp", "html", "css",
    "json", "yaml", "markdown", "plaintext",
]

# Text/code file extensions we can read and inject as context
TEXT_EXTS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css",
    ".json", ".yaml", ".yml", ".toml", ".md", ".txt",
    ".csv", ".sql", ".sh", ".bash", ".rs", ".go",
    ".java", ".c", ".cpp", ".h", ".hpp", ".rb", ".php",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

DEFAULT_SYSTEM = """\
You are an expert coding assistant. When you write code:
- Always wrap it in a markdown code block with the language tag (```python, ```javascript, etc.)
- Write complete, working code — not fragments
- Briefly explain what the code does

You have a code-runner tool. Use it to validate logic when helpful.\
"""

STARTER_CODE = "# Ask the agent to write code, or start coding here.\n"


# ─────────────────────────────────────────────────────────────────────
# Agentic tools
# ─────────────────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": (
                "Execute Python code in a sandboxed subprocess and return "
                "stdout + stderr. Use this to validate, test, or demonstrate code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to run (max ~50 lines, 5 s timeout).",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression precisely.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Python-compatible math expression, e.g. 'math.sqrt(2) * 100'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


# ─────────────────────────────────────────────────────────────────────
# Tool execution
# ─────────────────────────────────────────────────────────────────────
def _run_python(code: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        r = sp.run(
            ["python3", tmp],
            capture_output=True, text=True, timeout=5,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:3000] if out else "(no output)"
    except sp.TimeoutExpired:
        return "⏱ Timed out (>5 s)"
    except Exception as e:
        return f"Error: {e}"
    finally:
        os.unlink(tmp)


def _calculate(expr: str) -> str:
    ns = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    ns.update({"abs": abs, "round": round})
    try:
        return str(eval(expr, {"__builtins__": {}}, ns))  # noqa: S307
    except Exception as e:
        return f"Error: {e}"


def execute_tool(name: str, args: dict) -> str:
    if name == "run_code":
        return _run_python(args.get("code", ""))
    if name == "calculate":
        return _calculate(args.get("expression", ""))
    return f"Unknown tool: {name}"


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def encode_image(path: str) -> str | None:
    if not path:
        return None
    try:
        with open(path, "rb") as f:
            return b64.b64encode(f.read()).decode()
    except Exception:
        return None


def file_as_context(path: str) -> str | None:
    if not path:
        return None
    p = Path(path)
    if p.suffix.lower() not in TEXT_EXTS:
        return None
    try:
        content = p.read_text(encoding="utf-8", errors="replace")[:8000]
        lang = p.suffix.lstrip(".")
        return f"\n\n**Attached file — `{p.name}`:**\n```{lang}\n{content}\n```"
    except Exception:
        return None


def resolve_gradio_path(val) -> str | None:
    if val is None:
        return None

    if isinstance(val, Path):
        s = str(val)
        return s if s.strip() else None

    if isinstance(val, str):
        s = val.strip()
        return s if s else None

    if isinstance(val, dict):
        p = val.get("path")
        if isinstance(p, str) and p.strip():
            return p.strip()
        nested = val.get("file")
        if isinstance(nested, dict):
            np = nested.get("path")
            if isinstance(np, str) and np.strip():
                return np.strip()
        return None

    name = getattr(val, "name", None)

    if isinstance(name, str) and name.strip():
        return name.strip()

    return None


def is_image_path(path: str | None) -> bool:
    return bool(path) and Path(path).suffix.lower() in IMAGE_EXTS


def extract_last_code_block(text: str) -> tuple[str | None, str]:
    blocks = re.findall(r"```(\w*)\n(.*?)```", text, re.DOTALL)

    if blocks:
        lang, code = blocks[-1]
        return code.strip(), lang.strip() or "python"

    return None, "python"


def ollama_ok() -> bool:
    try:
        # print(f"Checking connection to Ollama... {OLLAMA_BASE}\n")
        res = r.get(f"{OLLAMA_BASE}/", timeout=30)
        if res.status_code == 200:
            return True
        return False
    except Exception as e:
        print(f'Got exception {e}')
        return False


def _gradio_content_to_text(content) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []

        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif "text" in block:
                    parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def _append_chat_turn(
            history: list | None, user_text: str, assistant_text: str
        ) -> list:
    base = list(history) if history else []

    return base + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


# ─────────────────────────────────────────────────────────────────────
# Run code (button handler)
# ─────────────────────────────────────────────────────────────────────
def run_code_btn(code: str) -> str:
    if not code.strip():
        return "Nothing to run."

    result = _run_python(code)
    return result


# ─────────────────────────────────────────────────────────────────────
# Core chat — streaming generator
# ─────────────────────────────────────────────────────────────────────
def chat(
            message: str,
            history: list,
            image_path: str | None,
            file_path: str | None,
            editor_code: str,
            language: str,
            system_prompt: str,
            agentic: bool,
            thinking: bool,
            temperature: float,
        ):
    img_for_vision = image_path if is_image_path(image_path) else (
        file_path if is_image_path(file_path) else None
    )

    has_text = bool(message.strip())
    has_attachment = bool(
        (file_path and not is_image_path(file_path))
        or img_for_vision
    )

    if not has_text and not has_attachment:
        yield history, editor_code, message
        return

    if not has_text and has_attachment:
        message = (
            "Use the attached file(s) as context. Summarize, answer questions, "
            "or improve the code as appropriate."
        )

    if not ollama_ok():
        err = "**Ollama is not running.** Start it with `ollama serve` and try again."
        yield _append_chat_turn(history, message, err), editor_code, ""
        return

    messages: list[dict] = []

    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})

    for h in history or []:
        if not isinstance(h, dict):
            continue
        role = h.get("role")
        if role not in ("user", "assistant"):
            continue
        messages.append(
            {"role": role, "content": _gradio_content_to_text(h.get("content"))}
        )

    content = message

    if editor_code.strip() and editor_code.strip() != STARTER_CODE.strip():
        content += (
            f"\n\n**Current code in editor ({language}):**\n"
            f"```{language}\n{editor_code}\n```"
        )

    if file_path and not is_image_path(file_path):
        ctx = file_as_context(file_path)
        if ctx:
            content += ctx
        else:
            fname = Path(file_path).name
            content += (
                f"\n\n*(Attached `{fname}` — extension not in the readable text/code "
                "list; paste its contents in chat if the model should see it.)*\n"
            )

    user_turn: dict = {"role": "user", "content": content}
    encoded = encode_image(img_for_vision)

    if encoded:
        user_turn["images"] = [encoded]

    messages.append(user_turn)
    options: dict = {"temperature": temperature}

    if thinking:
        options["think"] = True

    payload: dict = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "options": options,
    }

    if agentic:
        payload["tools"] = TOOLS

    try:
        resp = r.post(
            f"{OLLAMA_BASE}/api/chat",
            json=payload, stream=True, timeout=120,
        )

        resp.raise_for_status()
    except r.RequestException as e:
        yield _append_chat_turn(history, message, f"Request error: {e}"), editor_code, ""
        return

    full_response   = ""
    tool_logs       = []
    new_code        = editor_code

    for raw in resp.iter_lines():
        if not raw:
            continue
        try:
            chunk = json.loads(raw)
        except json.JSONDecodeError:
            continue

        msg   = chunk.get("message", {})
        piece = msg.get("content", "")

        if agentic and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn      = tc.get("function", {})
                fn_name = fn.get("name", "")
                fn_args = fn.get("arguments", {})
                result = execute_tool(fn_name, fn_args)
                tool_logs.append(
                    f"\n> **🔧 `{fn_name}`**\n"
                    f"> ```\n> {result.strip()}\n> ```"
                )

                if fn_name == "run_code" and fn_args.get("code"):
                    new_code = fn_args["code"]

                messages.append(msg)
                messages.append({"role": "tool", "content": result})

            try:
                resp2 = r.post(
                    f"{OLLAMA_BASE}/api/chat",
                    json={
                        "model": MODEL,
                        "messages": messages,
                        "stream": True,
                        "options": options,
                    },
                    stream=True, timeout=120,
                )

                resp2.raise_for_status()
            except r.RequestException as e:
                yield (
                    _append_chat_turn(history, message, f"Follow-up error: {e}"),
                    new_code,
                    "",
                )
                return

            tool_prefix = "\n".join(tool_logs) + "\n\n---\n\n"

            for raw2 in resp2.iter_lines():
                if not raw2:
                    continue
                try:
                    c2 = json.loads(raw2)
                except json.JSONDecodeError:
                    continue
                piece2 = c2.get("message", {}).get("content", "")
                full_response += piece2
                yield _append_chat_turn(
                    history, message, tool_prefix + full_response
                ), new_code, ""
            break
        full_response += piece
        yield _append_chat_turn(history, message, full_response), new_code, ""

    extracted, _ = extract_last_code_block(full_response)

    if extracted:
        new_code = extracted

    yield _append_chat_turn(history, message, full_response), new_code, ""


# ─────────────────────────────────────────────────────────────────────
# Respond wrapper (clears message input on first yield)
# ─────────────────────────────────────────────────────────────────────
def respond(
            message, history, image, file,
            code, lang, sys_p, agentic, thinking, temp,
        ):
    image_path = resolve_gradio_path(image)
    file_path = resolve_gradio_path(file)
    for h, c, _ in chat(
        message, history, image_path, file_path,
        code, lang, sys_p, agentic, thinking, temp,
    ):
        yield h, c, ""   # "" clears the message input


# ─────────────────────────────────────────────────────────────────────
# Theme-aware CSS — light: high contrast; dark: lifted text & borders
# (Previously all rules forced dark colors, so “light mode” still looked muddy.)
# ─────────────────────────────────────────────────────────────────────
CSS = """
/* ── Gradio variables: LIGHT (default) ── */
gradio-app {
    --body-background-fill:                   #f0f3f7;
    --block-background-fill:                  #ffffff;
    --border-color-primary:                   #c9d1d9;
    --border-color-accent:                    #8c959f;
    --body-text-color:                        #1f2328;
    --body-text-color-subdued:                #57606a;
    --block-label-text-color:                 #424a53;
    --input-background-fill:                  #ffffff;
    --input-border-color:                     #8c959f;
    --button-primary-background-fill:         #1a7f37;
    --button-primary-background-fill-hover:   #116329;
    --button-primary-text-color:              #ffffff;
    --button-secondary-background-fill:       #f6f8fa;
    --button-secondary-background-fill-hover: #eaeef2;
    --button-secondary-text-color:            #24292f;
    --button-secondary-border-color:          #8c959f;
    --color-accent:                           #0969da;
    --shadow-drop:                            0 1px 3px rgba(31, 35, 40, 0.12);
}

/* ── Gradio variables: DARK (higher contrast than before) ── */
body.dark gradio-app {
    --body-background-fill:                   #0d1117;
    --block-background-fill:                  #1c2128;
    --border-color-primary:                   #3d444d;
    --border-color-accent:                    #4d5560;
    --body-text-color:                        #e6edf3;
    --body-text-color-subdued:                #9da7b3;
    --block-label-text-color:                 #b1bac4;
    --input-background-fill:                  #22272e;
    --input-border-color:                     #4d5560;
    --button-primary-background-fill:         #238636;
    --button-primary-background-fill-hover:   #2ea043;
    --button-primary-text-color:              #ffffff;
    --button-secondary-background-fill:       #2d333b;
    --button-secondary-background-fill-hover: #373e47;
    --button-secondary-text-color:            #e6edf3;
    --button-secondary-border-color:          #4d5560;
    --color-accent:                           #58a6ff;
    --shadow-drop:                            none;
}

/* ── Layout ── */
body, .gradio-container {
    background: #f0f3f7 !important;
    max-width: 100% !important;
    padding: 0 !important;
}
body.dark, body.dark .gradio-container {
    background: #0d1117 !important;
}

/* ── Header ── */
#app-header {
    padding: 10px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #ffffff;
    border-bottom: 1px solid #c9d1d9;
}
body.dark #app-header {
    background: #1c2128;
    border-bottom-color: #3d444d;
}
.app-title    { font-size: 1.2rem; font-weight: 700; color: #1f2328; }
.app-subtitle { color: #424a53; font-size: 0.85rem; }
.app-badge {
    background: rgba(9, 105, 218, 0.12);
    border: 1px solid rgba(9, 105, 218, 0.35);
    color: #0550ae;
    border-radius: 10px;
    padding: 2px 10px;
    font-size: 0.72rem;
}
body.dark .app-title    { color: #e6edf3; }
body.dark .app-subtitle { color: #9da7b3; }
body.dark .app-badge {
    background: rgba(88, 166, 255, 0.12);
    border-color: rgba(88, 166, 255, 0.4);
    color: #79c0ff;
}

/* ── Panels / groups ── */
.gr-group, .gr-box {
    background: #ffffff !important;
    border: 1px solid #c9d1d9 !important;
    border-radius: 8px !important;
}
body.dark .gr-group, body.dark .gr-box {
    background: #1c2128 !important;
    border-color: #3d444d !important;
}

/* ── Code editor ── */
.cm-editor  { background: #ffffff !important; }
.cm-content { color: #24292f !important; font-size: 13px !important; }
.cm-gutters {
    background: #f6f8fa !important;
    border-right: 1px solid #d0d7de !important;
    color: #57606a !important;
    min-width: 36px;
}
.cm-activeLine       { background: #e7effb !important; }
.cm-activeLineGutter { background: #e7effb !important; }
.cm-selectionBackground { background: #add6ff !important; }
body.dark .cm-editor  { background: #0d1117 !important; }
body.dark .cm-content { color: #e6edf3 !important; }
body.dark .cm-gutters {
    background: #1c2128 !important;
    border-right-color: #3d444d !important;
    color: #7d8590 !important;
}
body.dark .cm-activeLine       { background: #21262d !important; }
body.dark .cm-activeLineGutter { background: #21262d !important; }
body.dark .cm-selectionBackground { background: #264f78 !important; }

/* ── Chatbot ── */
#chatbot {
    background: #f6f8fa !important;
    border: 1px solid #c9d1d9 !important;
    border-radius: 8px !important;
}
body.dark #chatbot {
    background: #0d1117 !important;
    border-color: #3d444d !important;
}
.message-wrap { padding: 4px 0 !important; }
.message.user {
    background: #ddf4ff !important;
    border: 1px solid #54aeff99 !important;
    border-radius: 8px !important;
}
.message.bot {
    background: #ffffff !important;
    border: 1px solid #c9d1d9 !important;
    border-radius: 8px !important;
}
body.dark .message.user { background: #21262d !important; border-color: #4d5560 !important; }
body.dark .message.bot  { background: #1c2128 !important; border-color: #3d444d !important; }

/* ── Inputs ── */
textarea, input[type="text"] {
    background: #ffffff !important;
    border: 1px solid #8c959f !important;
    color: #1f2328 !important;
    border-radius: 6px !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #0969da !important;
    box-shadow: 0 0 0 3px rgba(9, 105, 218, 0.18) !important;
}
textarea::placeholder { color: #6e7781 !important; }
body.dark textarea, body.dark input[type="text"] {
    background: #22272e !important;
    border-color: #4d5560 !important;
    color: #e6edf3 !important;
}
body.dark textarea:focus, body.dark input[type="text"]:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.2) !important;
}
body.dark textarea::placeholder { color: #7d8590 !important; }

/* ── Run output ── */
#run-output textarea {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 12px !important;
    color: #116329 !important;
    background: #dafbe1 !important;
    border: 1px solid #2da44e66 !important;
}
body.dark #run-output textarea {
    color: #3fb950 !important;
    background: #0d1117 !important;
    border-color: #30363d !important;
}

/* ── Labels ── */
label > span:first-child {
    color: #424a53 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
    font-weight: 600 !important;
}
body.dark label > span:first-child { color: #b1bac4 !important; }

/* ── Run button (blue) ── */
.run-btn {
    background: #0969da !important;
    border: 1px solid #0550ae !important;
    color: #ffffff !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}
.run-btn:hover { background: #0550ae !important; }
body.dark .run-btn {
    background: #1f6feb !important;
    border-color: #388bfd !important;
}
body.dark .run-btn:hover { background: #388bfd !important; }

/* ── Code blocks inside chat ── */
code {
    background: #f6f8fa !important;
    color: #0550ae !important;
    border: 1px solid #d0d7de !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
    font-size: 0.85em !important;
}
pre {
    background: #f6f8fa !important;
    border: 1px solid #d0d7de !important;
    border-radius: 6px !important;
    padding: 12px !important;
}
pre code { background: transparent !important; border: none !important; padding: 0 !important; }
body.dark code {
    background: #21262d !important;
    color: #79c0ff !important;
    border-color: #3d444d !important;
}
body.dark pre {
    background: #21262d !important;
    border-color: #3d444d !important;
}

/* ── Accordion ── */
.gr-accordion {
    background: #ffffff !important;
    border: 1px solid #c9d1d9 !important;
    border-radius: 8px !important;
}
.gr-accordion > .label-wrap { color: #424a53 !important; font-weight: 600 !important; }
body.dark .gr-accordion { background: #1c2128 !important; border-color: #3d444d !important; }
body.dark .gr-accordion > .label-wrap { color: #b1bac4 !important; }

/* ── Checkboxes ── */
input[type="checkbox"] { accent-color: #0969da !important; }
body.dark input[type="checkbox"] { accent-color: #58a6ff !important; }

/* ── Dropdown ── */
.gr-dropdown {
    background: #ffffff !important;
    border-color: #8c959f !important;
    color: #1f2328 !important;
}
body.dark .gr-dropdown {
    background: #22272e !important;
    border-color: #4d5560 !important;
    color: #e6edf3 !important;
}

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #eaeef2; }
::-webkit-scrollbar-thumb { background: #afb8c1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #8c959f; }
body.dark ::-webkit-scrollbar-track { background: #0d1117; }
body.dark ::-webkit-scrollbar-thumb { background: #4d5560; }
body.dark ::-webkit-scrollbar-thumb:hover { background: #6e7781; }

/* ── Upload controls: keep native file picker usable ── */
label.wrap input[type="file"],
.wrap input[type="file"] {
    cursor: pointer !important;
    min-height: 2rem;
}

/* ── Remove Gradio footer ── */
footer { display: none !important; }
"""

# ─────────────────────────────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────────────────────────────
THEME = gr.themes.Base(
    primary_hue   = "blue",
    secondary_hue = "slate",
    neutral_hue   = "slate",
    font      = [gr.themes.GoogleFont("Inter"),           "system-ui", "sans-serif"],
    font_mono = [gr.themes.GoogleFont("JetBrains Mono"),  "monospace"],
)


# ─────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Gemma 4 · Code Assistant") as demo:

    # ── Header ────────────────────────────────────────────────────────
    gr.HTML("""
        <div id="app-header">
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
            <span class="app-title">Gemma 4</span>
            <span class="app-subtitle">Code Assistant</span>
            <span class="app-badge">e4b · local</span>
        </div>
        </div>
        """)

    # ── Main split pane ───────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # ── LEFT: Code Editor ─────────────────────────────────────────
        with gr.Column(scale=11):

            # Toolbar
            with gr.Row():
                lang_sel = gr.Dropdown(
                    choices=LANGUAGES, value="python",
                    label="Language", scale=3, container=True, min_width=130,
                )
                run_btn   = gr.Button("Run Code",   elem_classes=["run-btn"],  scale=2, min_width=90)
                clear_ed  = gr.Button(" Clear", variant="secondary",       scale=2, min_width=90)

            # Editor
            code_editor = gr.Code(
                value=STARTER_CODE,
                language="python",
                label="Editor",
                lines=24,
                interactive=True,
            )

            # Execution output
            run_output = gr.Textbox(
                value="",
                label="Output",
                lines=6,
                max_lines=12,
                interactive=False,
                elem_id="run-output",
                placeholder="Execution output will appear here…",
            )

        # ── RIGHT: Chat ───────────────────────────────────────────────
        with gr.Column(scale=9):

            # Chat history
            chatbot = gr.Chatbot(
                value=[],
                elem_id="chatbot",
                label="Chat",
                height=430,
            )

            # Attachments
            with gr.Row():
                image_upload = gr.Image(
                    label="Image (vision)",
                    type="filepath",
                    height=200,
                    min_width=200,
                    # None = upload + webcam + clipboard; explicit upload avoids picker quirks in some builds
                    sources=["upload", "webcam", "clipboard"],
                    scale=1,
                )
                file_upload = gr.File(
                    label="Code / text file",
                    height=140,
                    scale=1,
                    file_count="single",
                    # Allow any file in the OS picker; only known text/code extensions are read into context
                    file_types=None,
                )

            # Message input + send
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask the agent to write, explain, debug or improve code…",
                    show_label=False,
                    scale=6,
                    lines=1,
                    max_lines=6,
                    container=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1, min_width=70)

            # Controls row
            with gr.Row():
                agentic_cb  = gr.Checkbox(label="Enable Agentic",  value=True,  scale=1)
                thinking_cb = gr.Checkbox(label="Enable Thinking", value=False, scale=1)
                clear_chat  = gr.Button("Clear chat", variant="secondary", scale=1, min_width=100)

            # Advanced settings (collapsed by default)
            with gr.Accordion("Settings", open=False):
                sys_prompt = gr.Textbox(
                    value=DEFAULT_SYSTEM,
                    label="System prompt",
                    lines=4,
                    max_lines=10,
                )
                temperature = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.7, step=0.05,
                    label="Temperature",
                )

    # ─────────────────────────────────────────────────────────────────
    # Event wiring
    # ─────────────────────────────────────────────────────────────────

    # Run code
    run_btn.click(fn=run_code_btn, inputs=[code_editor], outputs=[run_output])

    # Clear editor
    clear_ed.click(fn=lambda: (STARTER_CODE, ""), outputs=[code_editor, run_output])

    # Language selector → update editor syntax highlight
    lang_sel.change(
        fn=lambda lang: gr.update(language=lang),
        inputs=[lang_sel],
        outputs=[code_editor],
    )

    # Shared inputs/outputs for send
    _inputs  = [
        msg_input, chatbot, image_upload, file_upload,
        code_editor, lang_sel,
        sys_prompt, agentic_cb, thinking_cb, temperature,
    ]

    _outputs = [chatbot, code_editor, msg_input]

    send_btn.click(fn=respond, inputs=_inputs, outputs=_outputs)
    msg_input.submit(fn=respond, inputs=_inputs, outputs=_outputs)

    # Clear chat
    clear_chat.click(
        fn=lambda: ([], None, None, ""),
        outputs=[chatbot, image_upload, file_upload, run_output],
    )


# ─────────────────────────────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n Gemma 4 Code Assistant")
    print(f"    Model  : {MODEL}")
    print(f"    Ollama : {OLLAMA_BASE}")

    if not ollama_ok():
        print(" Ollama not detected — run `ollama serve` before chatting")

    print("    UI     : http://localhost:7860\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=THEME,
        css=CSS,
    )
