"""Two deciders: Jev (typed choice) and Astra (tool calls).

Jev answers a typed question with probabilities and a confidence value. Astra is
a general model: it sees the camera images and chooses a skill by name.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np
import numpy.typing as npt
from PIL import Image

OPENROUTER_DECISIONS = "https://openrouter.ai/api/alpha/decisions"
OPENAI_RESPONSES = "https://api.openai.com/v1/responses"


def _png_b64(rgb: npt.NDArray[np.uint8], size: int = 512) -> str:
    img = Image.fromarray(rgb)
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@dataclass
class Decision:
    """What to do next."""

    skill: str
    confidence: float | None = None
    note: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Jev:
    """TypeSafe Jev through OpenRouter. Text state in, typed choice out."""

    model: str = "~typesafe/jev-latest"
    calls: int = 0
    seconds: float = 0.0
    cost: float = 0.0

    def choose(self, state: dict[str, Any], skills: dict[str, str],
               instructions: str) -> Decision:
        body = {
            "model": self.model,
            "state": json.dumps(state, indent=1),
            "questions": {"next_skill": {"type": "choice", "instructions": instructions,
                                         "criteria": skills}},
        }
        t0 = time.time()
        r = httpx.post(OPENROUTER_DECISIONS, json=body, timeout=60,
                       headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
        r.raise_for_status()
        data = r.json()
        self.calls += 1
        self.seconds += time.time() - t0
        self.cost += float(data.get("usage", {}).get("cost", 0.0) or 0.0)
        ans = data["answers"]["next_skill"]
        return Decision(ans["choice"], ans.get("confidence"), raw=ans)


@dataclass
class Astra:
    """OpenAI GPT-6 Astra. Sees the images, picks a skill and says why."""

    model: str = "gpt-6-astra"
    effort: str = "high"
    calls: int = 0
    seconds: float = 0.0
    history: list[str] = field(default_factory=list)

    def choose(self, state: dict[str, Any], skills: dict[str, str], instructions: str,
               images: dict[str, npt.NDArray[np.uint8]]) -> Decision:
        schema = {"type": "object", "additionalProperties": False,
                  "properties": {"skill": {"type": "string", "enum": list(skills)},
                                 "note": {"type": "string"}},
                  "required": ["skill", "note"]}
        text = (f"{instructions}\n\nSkills:\n"
                + "\n".join(f"- {k}: {v}" for k, v in skills.items())
                + f"\n\nState:\n{json.dumps(state, indent=1)}\n"
                + "Recent decisions:\n" + "\n".join(self.history[-6:])
                + "\n\nPick one skill. In `note`, say in one or two sentences what you see "
                  "and why you chose it.")
        content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
        for name, rgb in images.items():
            content.append({"type": "input_text", "text": f"camera '{name}':"})
            content.append({"type": "input_image",
                            "image_url": f"data:image/png;base64,{_png_b64(rgb)}"})
        body = {"model": self.model, "reasoning": {"effort": self.effort},
                "input": [{"role": "user", "content": content}],
                "text": {"format": {"type": "json_schema", "name": "decision",
                                    "schema": schema, "strict": True}}}
        t0 = time.time()
        r = httpx.post(OPENAI_RESPONSES, json=body, timeout=600,
                       headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
        r.raise_for_status()
        data = r.json()
        self.calls += 1
        self.seconds += time.time() - t0
        out = "".join(c.get("text", "") for item in data.get("output", [])
                      if item.get("type") == "message" for c in item.get("content", []))
        parsed = json.loads(out)
        self.history.append(f"{parsed['skill']}: {parsed['note']}")
        return Decision(parsed["skill"], None, parsed["note"], parsed)


def median_ms(values: list[float]) -> float:
    return float(np.median(values) * 1000) if values else float("nan")
