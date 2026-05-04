from __future__ import annotations

import html
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

from .data import DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH
from .model import PredictionResult
from .train import ensure_model


ResponseBody = list[bytes]
StartResponse = Callable[[str, list[tuple[str, str]]], None]


class FakeNewsWebApp:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, data_path: Path = DEFAULT_DATA_PATH) -> None:
        self.detector = ensure_model(model_path=model_path, data_path=data_path)

    def __call__(self, environ: dict, start_response: StartResponse) -> ResponseBody:
        path = environ.get("PATH_INFO", "/")
        method = environ.get("REQUEST_METHOD", "GET").upper()

        if path == "/health":
            start_response("200 OK", [("Content-Type", "text/plain; charset=utf-8")])
            return [b"ok"]

        if path == "/" and method == "GET":
            start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
            return [self.render_page().encode("utf-8")]

        if path == "/predict" and method == "POST":
            form_data = self._read_form_data(environ)
            article_text = form_data.get("article_text", [""])[0].strip()
            if not article_text:
                start_response("400 Bad Request", [("Content-Type", "text/html; charset=utf-8")])
                return [self.render_page(error="Paste a headline or article before analyzing.").encode("utf-8")]

            result = self.detector.predict(article_text)
            start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
            return [self.render_page(article_text=article_text, result=result).encode("utf-8")]

        start_response("404 Not Found", [("Content-Type", "text/plain; charset=utf-8")])
        return [b"Not Found"]

    def _read_form_data(self, environ: dict) -> dict[str, list[str]]:
        length = int(environ.get("CONTENT_LENGTH") or 0)
        body = environ["wsgi.input"].read(length).decode("utf-8") if length else ""
        return parse_qs(body, keep_blank_values=True)

    def render_page(
        self,
        article_text: str = "",
        result: PredictionResult | None = None,
        error: str | None = None,
    ) -> str:
        escaped_text = html.escape(article_text)
        escaped_error = html.escape(error) if error else ""
        result_section = self._render_result(result) if result else ""
        error_section = (
            f'<p class="notice error">{escaped_error}</p>' if escaped_error else ""
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Signal Desk | Fake News Detector</title>
  <style>
    :root {{
      --paper: #f4efe3;
      --paper-deep: #e6dcc8;
      --ink: #1f1a17;
      --muted: #665c55;
      --line: rgba(31, 26, 23, 0.14);
      --accent: #8f2d1d;
      --accent-soft: rgba(143, 45, 29, 0.12);
      --real: #1c6b49;
      --real-soft: rgba(28, 107, 73, 0.12);
      --fake: #a12a1b;
      --fake-soft: rgba(161, 42, 27, 0.12);
      --shadow: 0 24px 60px rgba(37, 27, 18, 0.14);
      --headline-font: Georgia, "Times New Roman", serif;
      --body-font: "Trebuchet MS", "Segoe UI", sans-serif;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      color: var(--ink);
      font-family: var(--body-font);
      background:
        radial-gradient(circle at top, rgba(255, 255, 255, 0.55), transparent 34%),
        repeating-linear-gradient(
          0deg,
          rgba(255, 255, 255, 0.18),
          rgba(255, 255, 255, 0.18) 2px,
          transparent 2px,
          transparent 5px
        ),
        linear-gradient(135deg, #d8cab2, var(--paper));
      min-height: 100vh;
    }}

    .page {{
      width: min(1080px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0 48px;
    }}

    .masthead {{
      display: grid;
      gap: 12px;
      margin-bottom: 28px;
    }}

    .eyebrow {{
      letter-spacing: 0.16em;
      text-transform: uppercase;
      font-size: 0.8rem;
      color: var(--muted);
    }}

    h1 {{
      margin: 0;
      font-family: var(--headline-font);
      font-size: clamp(2.4rem, 4vw, 4.7rem);
      line-height: 0.95;
      max-width: 12ch;
    }}

    .lede {{
      max-width: 62ch;
      font-size: 1.05rem;
      line-height: 1.7;
      color: var(--muted);
      margin: 0;
    }}

    .grid {{
      display: grid;
      gap: 24px;
      grid-template-columns: 1.15fr 0.85fr;
      align-items: start;
    }}

    .panel {{
      background: rgba(255, 252, 246, 0.8);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}

    .panel h2 {{
      margin: 0 0 10px;
      font-family: var(--headline-font);
      font-size: 1.7rem;
    }}

    .panel p {{
      color: var(--muted);
      line-height: 1.65;
      margin-top: 0;
    }}

    textarea {{
      width: 100%;
      min-height: 280px;
      resize: vertical;
      padding: 18px;
      border-radius: 18px;
      border: 1px solid var(--line);
      font: inherit;
      line-height: 1.6;
      color: var(--ink);
      background: rgba(255, 255, 255, 0.74);
    }}

    textarea:focus {{
      outline: 2px solid rgba(143, 45, 29, 0.25);
      border-color: rgba(143, 45, 29, 0.4);
    }}

    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      align-items: center;
      margin-top: 16px;
    }}

    button {{
      border: 0;
      border-radius: 999px;
      background: linear-gradient(135deg, #271915, var(--accent));
      color: white;
      font: inherit;
      font-weight: 700;
      padding: 13px 22px;
      cursor: pointer;
      box-shadow: 0 12px 26px rgba(143, 45, 29, 0.22);
      transition: transform 0.18s ease, box-shadow 0.18s ease;
    }}

    button:hover {{
      transform: translateY(-1px);
      box-shadow: 0 16px 32px rgba(143, 45, 29, 0.28);
    }}

    .hint {{
      font-size: 0.94rem;
      color: var(--muted);
    }}

    .notice {{
      padding: 12px 14px;
      border-radius: 14px;
      font-size: 0.95rem;
      margin: 0 0 14px;
    }}

    .error {{
      background: rgba(161, 42, 27, 0.08);
      color: var(--fake);
      border: 1px solid rgba(161, 42, 27, 0.16);
    }}

    .result-card {{
      display: grid;
      gap: 18px;
    }}

    .result-banner {{
      border-radius: 20px;
      padding: 18px;
    }}

    .result-banner.fake {{
      background: linear-gradient(135deg, var(--fake-soft), rgba(255, 255, 255, 0.45));
      border: 1px solid rgba(161, 42, 27, 0.18);
    }}

    .result-banner.real {{
      background: linear-gradient(135deg, var(--real-soft), rgba(255, 255, 255, 0.45));
      border: 1px solid rgba(28, 107, 73, 0.18);
    }}

    .label {{
      display: inline-block;
      font-size: 0.78rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }}

    .verdict {{
      margin: 0;
      font-family: var(--headline-font);
      font-size: 2.15rem;
    }}

    .confidence {{
      margin: 8px 0 0;
      color: var(--muted);
    }}

    .stats {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}

    .stat {{
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      background: rgba(255, 255, 255, 0.62);
    }}

    .stat strong {{
      display: block;
      font-size: 1.4rem;
      margin-top: 4px;
      font-family: var(--headline-font);
    }}

    ul {{
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
      line-height: 1.7;
    }}

    .footnote {{
      font-size: 0.92rem;
      color: var(--muted);
      margin-top: 18px;
    }}

    @media (max-width: 860px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}

      .page {{
        width: min(100% - 20px, 1080px);
        padding-top: 20px;
      }}

      textarea {{
        min-height: 220px;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="masthead">
      <div class="eyebrow">Signal Desk Prototype</div>
      <h1>Fake-news screening for headlines and article snippets.</h1>
      <p class="lede">Paste a claim, headline, or short article below. The detector compares the wording against a small trained corpus and highlights the phrases that pushed the prediction. It is useful as a first-pass screening tool, not a substitute for real source verification.</p>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>Analyze text</h2>
        <p>Try a dramatic social post, a reported article excerpt, or a headline you want to triage quickly.</p>
        {error_section}
        <form method="post" action="/predict">
          <textarea name="article_text" placeholder="Paste a headline or short article here...">{escaped_text}</textarea>
          <div class="toolbar">
            <button type="submit">Scan for signals</button>
            <span class="hint">Tip: longer snippets with source language usually produce steadier results.</span>
          </div>
        </form>
      </div>

      <aside class="panel">
        <h2>Reading the result</h2>
        <p>The model looks for repeated word patterns, sensational phrasing, attribution language, and other surface-level signals. It does not fetch the web or verify a source database.</p>
        {result_section}
        <p class="footnote">Starter model training data lives in <code>data/seed_training_data.csv</code>. Retraining on a stronger dataset will improve the detector far more than tweaking the UI.</p>
      </aside>
    </section>
  </main>
</body>
</html>"""

    def _render_result(self, result: PredictionResult) -> str:
        verdict = "Likely fake" if result.label == "fake" else "Likely real"
        banner_class = "fake" if result.label == "fake" else "real"
        fake_score = f"{result.probabilities['fake'] * 100:.1f}%"
        real_score = f"{result.probabilities['real'] * 100:.1f}%"
        evidence_items = "".join(
            f"<li><strong>{html.escape(token)}</strong> leans {'fake' if score > 0 else 'real'}.</li>"
            for token, score in result.evidence
        ) or "<li>No strong keywords were found in the current vocabulary.</li>"

        return f"""
<section class="result-card">
  <div class="result-banner {banner_class}">
    <div class="label">Current verdict</div>
    <h3 class="verdict">{verdict}</h3>
    <p class="confidence">Confidence: {result.confidence * 100:.1f}%</p>
  </div>
  <div class="stats">
    <div class="stat">
      Fake signal
      <strong>{fake_score}</strong>
    </div>
    <div class="stat">
      Real signal
      <strong>{real_score}</strong>
    </div>
  </div>
  <div>
    <div class="label">Top language cues</div>
    <ul>{evidence_items}</ul>
  </div>
</section>"""


def main() -> None:
    host = "127.0.0.1"
    port = 8000
    app = FakeNewsWebApp()
    print(f"Starting Fake News Detector at http://{host}:{port}")
    with make_server(host, port, app) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
