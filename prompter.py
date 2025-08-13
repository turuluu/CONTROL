# prompt_eval_tui.py
# Python 3.12
# A Textual TUI to compare prompt wordings side-by-side with deterministic Ollama generations.
# - 3 columns, 1 result each, sequential streaming
# - Global locked sampling params (only wording varies)
# - Stage-based drop scoring with JSON saves in ~/.prompt-test-saves
# - Optional "Generate alternative" via LlamaIndex (separate, non-deterministic generator)
#
# Install (in a local .venv):
#   python3.12 -m venv .venv && source .venv/bin/activate
#   pip install --upgrade pip
#   pip install textual>=0.58 httpx>=0.27 rich>=13.7 pydantic>=2.7
#   pip install llama-index>=0.11 llama-index-llms-ollama>=0.2  # optional, for "generate alternative"
#
# Run:
#   python prompt_eval_tui.py
#
# Key combos:
#   1e/2e/3e = edit prompt in that column
#   1d/2d/3d = mark worst & drop that column (increments stage)
#   1g/2g/3g = generate alternative into that (empty) column (rewrites dropped prompt)
#   r = run all (sequential, streaming)
#   n = manually add new prompt into an empty slot
#   m = change globals (model/temp/seed/num_ctx/num_predict etc.)
#   s = save run        o = open run JSON
#   q = quit
#
# Notes:
# - Defaults: model='gpt-oss', deterministic temperature=0, fixed seed per run.
# - Uses direct Ollama /api/generate for deterministic control; LlamaIndex used only for "generate alternative".
# - No auto-pull: you manage models in Ollama.

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from rich.pretty import Pretty
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import Footer, Header, Input, Label, Static, Log, RichLog, TextArea

# NEW: a small modal screen for single/multi-line input
from textual.screen import ModalScreen
from textual.widgets import Button, TextArea, Input, Label
from textual.containers import Container, Horizontal

class PromptEditor(ModalScreen[Optional[str]]):
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("ctrl+s", "save", "Save"),
    ]
    def __init__(self, title: str, initial: str = "", multiline: bool = True) -> None:
        super().__init__()
        self.title_text = title
        self.initial = initial
        self.multiline = multiline
        self._input: Optional[Input] = None
        self._area: Optional[TextArea] = None

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_save(self) -> None:
        if self.multiline and self._area:
            self.dismiss(self._area.text)
        elif self._input:
            self.dismiss(self._input.value)
        else:
            self.dismiss("")

    def compose(self):
        with Container(id="box"):
            yield Label(self.title_text)
            if self.multiline:
                self._area = TextArea()
                yield self._area
            else:
                self._input = Input(self.initial)
                yield self._input
            with Horizontal():
                yield Button("Save", id="save", variant="primary")
                yield Button("Cancel", id="cancel")

    def on_mount(self):
        if self.multiline and self._area:
            self._area.text = self.initial
            self._area.focus()
        elif self._input:
            self._input.value = self.initial
            self._input.focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            if self.multiline and self._area:
                self.dismiss(self._area.text)
            elif self._input:
                self.dismiss(self._input.value)
            else:
                self.dismiss("")
        else:
            self.dismiss(None)

SAVE_DIR = Path(os.path.expanduser("~/.prompt-test-saves"))
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Data model
# -------------------------------

@dataclass
class GenStats:
    total_duration_ms: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration_ms: Optional[int] = None

@dataclass
class OutputRecord:
    timestamp: str
    text: str
    stats: GenStats = field(default_factory=GenStats)

@dataclass
class PromptSlot:
    slot_id: int
    text: str = ""
    active: bool = True           # false when dropped or empty
    empty: bool = True            # true if slot has no prompt yet
    dropped: bool = False
    born_stage: int = 0
    has_survived: bool = False    # flips true once it survives one drop after birth
    final_score: Optional[float] = None
    last_output: str = ""
    history: List[OutputRecord] = field(default_factory=list)
    last_dropped_text: Optional[str] = None   # remembers what was dropped (for generator)

    def current_score(self, stage: int) -> float:
        # Scoring rule:
        # - Global "stage" increments after each drop action.
        # - A prompt that has survived at least one drop while alive shows score = 1 / stage.
        # - A newly added prompt that hasn't yet survived a drop displays 1.0.
        # - When dropped at stage N, its final_score is 1 / N.
        if self.dropped and self.final_score is not None:
            return self.final_score
        if not self.empty and self.has_survived and stage > 0:
            return 1.0 / stage
        return 1.0

@dataclass
class RunSettings:
    model: str = "gpt-oss"
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repeat_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    num_ctx: int = 8192          # default context length
    num_predict: int = 1024      # default max tokens to generate per output
    seed: int = 42
    base_url: str = "http://localhost:11434"  # Ollama

@dataclass
class GeneratorSettings:
    # For "Generate alternative" only (non-deterministic, separate from evaluation runs)
    model: str = "gpt-oss"
    temperature: float = 0.7
    top_p: float = 0.95
    num_ctx: int = 8192
    num_predict: int = 512

@dataclass
class RunState:
    run_id: str
    created_at: str
    stage: int = 0                   # increments after each drop
    settings: RunSettings = field(default_factory=RunSettings)
    gen_settings: GeneratorSettings = field(default_factory=GeneratorSettings)
    slots: List[PromptSlot] = field(default_factory=list)
    note: str = ""

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "RunState":
        settings = RunSettings(**data["settings"])
        gens = GeneratorSettings(**data.get("gen_settings", {}))
        slots = []
        for s in data["slots"]:
            ps = PromptSlot(
                slot_id=s["slot_id"],
                text=s.get("text", ""),
                active=s.get("active", True),
                empty=s.get("empty", True),
                dropped=s.get("dropped", False),
                born_stage=s.get("born_stage", 0),
                has_survived=s.get("has_survived", False),
                final_score=s.get("final_score"),
                last_output=s.get("last_output", ""),
                history=[
                    OutputRecord(
                        timestamp=rec.get("timestamp", ""),
                        text=rec.get("text", ""),
                        stats=GenStats(
                            total_duration_ms=rec.get("stats", {}).get("total_duration_ms"),
                            eval_count=rec.get("stats", {}).get("eval_count"),
                            eval_duration_ms=rec.get("stats", {}).get("eval_duration_ms"),
                        ),
                    )
                    for rec in s.get("history", [])
                ],
                last_dropped_text=s.get("last_dropped_text"),
            )
            slots.append(ps)
        return RunState(
            run_id=data["run_id"],
            created_at=data["created_at"],
            stage=data.get("stage", 0),
            settings=settings,
            gen_settings=gens,
            slots=slots,
            note=data.get("note", ""),
        )

# -------------------------------
# UI widgets
# -------------------------------

class HeaderBar(Static):
    """Compact header showing globals & stage."""

    def __init__(self, state: RunState) -> None:
        super().__init__()

        self.state = state

    def refresh_bar(self) -> None:
        """Recompute and update the header content."""

        s = self.state.settings
        self.update(
            Pretty(
                {
                    "model": s.model,
                    "temp": s.temperature,
                    "seed": s.seed,
                    "num_ctx": s.num_ctx,
                    "num_predict": s.num_predict,
                    "top_p": s.top_p,
                    "top_k": s.top_k,
                    "repeat_penalty": s.repeat_penalty,
                    "stage": self.state.stage,
                },
                expand_all=True,
            )
        )

class PromptPanel(Static):
    """One column: prompt text + output log."""

    class Edited(Message):
        def __init__(self, slot_id: int, new_text: str) -> None:
            self.slot_id = slot_id
            self.new_text = new_text
            super().__init__()

    def __init__(self, slot: PromptSlot) -> None:
        super().__init__()
        self.slot = slot
        self.prompt_label = Label("", id=f"prompt-label-{slot.slot_id}")
        self.prompt_text = RichLog(id=f"prompt-text-{slot.slot_id}", highlight=False)
        self.output_log = RichLog(id=f"output-log-{slot.slot_id}", highlight=False)

    def compose(self) -> ComposeResult:
        yield Label(f"Prompt {self.slot.slot_id}", classes="title")
        yield self.prompt_text
        yield Label("Output", classes="title")
        yield self.output_log

    def refresh_prompt(self) -> None:
        self.prompt_text.clear()
        if self.slot.empty:
            self.prompt_text.write("[empty slot]  (use 'n' to add or '{id}g' to generate)".format(id=self.slot.slot_id))
        else:
            self.prompt_text.write(self.slot.text)

    def clear_output(self) -> None:
        self.output_log.clear()

    def append_output(self, text: str) -> None:
        self.output_log.write(text)

    def replace_output(self, text: str) -> None:
        self.output_log.clear()
        if text:
            self.output_log.write(text)

class SidePanel(VerticalScroll):
    """Shows scores and status for all prompts."""

    def __init__(self, state: RunState) -> None:
        super().__init__()
        self.state = state
        self.labels: Dict[int, Label] = {}
        self.status_log: Optional[RichLog] = None

    def compose(self) -> ComposeResult:
        yield Label("Prompts & Scores", classes="side-title")
        for s in self.state.slots:
            lbl = Label("", id=f"score-{s.slot_id}")
            self.labels[s.slot_id] = lbl
            yield lbl
        yield Label("\nHints: \n• '1e/2e/3e' edit \n• '1d/2d/3d' drop \n• '1g/2g/3g' alt \n• r run \n• n new \n• m globals \n• s save \n• o open \n• q quit", classes="hints")
        yield Label("\nStatus", classes="side-title")
        self.status_log = RichLog(id="status-log")
        yield self.status_log

    def update_panel(self) -> None:

        for s in self.state.slots:
            status = "EMPTY" if s.empty else ("DROPPED" if s.dropped else "ACTIVE")
            score = s.current_score(self.state.stage)
            text = f"[{s.slot_id}] {status}  score={score:.4f}"
            if not s.empty and not s.dropped:
                text += f"  born@{s.born_stage}  survived={s.has_survived}"
            self.labels[s.slot_id].update(text)

    def post_status(self, msg: str) -> None:
        if self.status_log is not None:
            self.status_log.write(msg)

class ModalInput(App):
    """Simple modal prompt editor—spawned as a sub-App (blocking)."""

    CSS = """
    Screen { align: center middle; }
    #box { width: 90%; height: 80%; border: solid $accent; padding: 1 2; }
    Input { width: 100%; }
    TextArea { height: 1fr; }
    """

    def __init__(self, title: str, initial: str = "", multiline: bool = True) -> None:
        super().__init__()
        self.title_text = title
        self.initial = initial
        self.multiline = multiline
        self.result: Optional[str] = None
        self.input_widget: Optional[Input] = None
        self.editor = TextArea()


    def compose(self) -> ComposeResult:
        with Container(id="box"):
            yield Label(self.title_text)
            if self.multiline:
                self.editor = TextArea()
                yield self.editor
            else:
                self.input_widget = Input(self.initial)
                yield self.input_widget
            yield Label("Press Ctrl+S to save, Esc to cancel.")

    def on_mount(self) -> None:
        if self.multiline and self.editor:
                self.editor.text = self.initial
        elif self.input_widget:
            self.input_widget.value = self.initial

    BINDINGS = [
        Binding("ctrl+s", "save", "Save"),
        Binding("escape", "cancel", "Cancel"),
    ]

    def action_save(self) -> None:
        if self.multiline and self.editor:
            self.result = self.editor.text
        elif self.input_widget:
            self.result = self.input_widget.value
        self.exit()

    def action_cancel(self) -> None:
        self.result = None
        self.exit()

# -------------------------------
# Core App
# -------------------------------

class PromptEvalApp(App):
    CSS = """
    Screen {
        layout: grid;
        grid-size: 4 2;
        grid-rows: auto 1fr;
        grid-columns: 28 1fr 1fr 1fr;
    }
    .title { content-align: left middle; padding: 0 1; }
    .side-title { padding: 1 1; }
    #side   { row-span: 2; border: tall $accent; }
    #header { column-span: 3; border: tall $accent; }
    #col1, #col2, #col3 { border: round $primary; }
    RichLog { height: 1fr; }
    #status-log { height: 10; }  /* optional: give status a compact fixed height */
    """

    BINDINGS = [
        Binding("r", "run_all", "Run"),
        Binding("n", "add_new", "New"),
        Binding("m", "edit_globals", "Globals"),
        Binding("s", "save_run", "Save"),
        Binding("o", "open_run", "Open"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.state = self._new_run_state()
        self.header_bar = HeaderBar(self.state)
        self.side_panel = SidePanel(self.state)
        self.panels: Dict[int, PromptPanel] = {}
        self.combo_buffer: Optional[int] = None
        self.combo_ts: float = 0.0
        self.client: Optional[httpx.AsyncClient] = None
        self.run_lock = asyncio.Lock()

    def _new_run_state(self) -> RunState:
        return RunState(
            run_id=str(uuid.uuid4()),
            created_at=datetime.now(UTC).isoformat(),
            stage=0,
            settings=RunSettings(),
            gen_settings=GeneratorSettings(),
            slots=[PromptSlot(slot_id=i+1, empty=True) for i in range(3)],
        )

    # -------------- Lifecycle

    def compose(self) -> ComposeResult:
        # Order matters for grid placement.
        self.side_panel.id = "side"
        yield self.side_panel  # (1,1) spanning two rows
        self.header_bar.id = "header"
        yield self.header_bar  # (1,2) spanning 3 columns
        # Three prompt columns on row 2, columns 2..4
        for i in range(1, 4):
            panel = PromptPanel(self.state.slots[i - 1])
            panel.id = f"col{i}"
            self.panels[i] = panel
            yield panel

    async def on_mount(self) -> None:
        # Side effects only (no layout API calls here on latest Textual).
        self.client = httpx.AsyncClient(timeout=30.0)
        self.refresh_all()

    async def on_unmount(self) -> None:
        if self.client:
            await self.client.aclose()

    # -------------- UI updates

    def refresh_all(self) -> None:
        self.header_bar.refresh_bar()
        self.side_panel.update_panel()
        for i, panel in self.panels.items():
            panel.refresh_prompt()
            if self.state.slots[i-1].last_output:
                panel.replace_output(self.state.slots[i-1].last_output)

    # -------------- Key handling for combos like "2e", "3d", "1g"

    async def on_key(self, event) -> None:
        key = event.key
        now = time.time()

        if key == "r":
            await self.action_run_all()
            return

        if key in ("1", "2", "3"):
            self.combo_buffer = int(key)
            self.combo_ts = now
            return

        if self.combo_buffer and (now - self.combo_ts) < 1.2:
            slot = self.combo_buffer
            self.combo_buffer = None

            if key == "e":
                await self._edit_prompt(slot); return
            if key == "d":
                await self._drop_slot(slot); return
            if key == "g":
                await self._generate_alternative(slot); return
            if key == "n":
                self.run_worker(self._add_new_in_slot(slot), exclusive=True)
                return
        # fall through to normal single-key bindings

    # -------------- Actions

    async def action_run_all(self) -> None:
        if self.run_lock.locked():
            await self._status("Already running…")
            return
        async with self.run_lock:
            # Sequential runs across active, non-empty slots (1->3)
            await self._status(f"Starting run… model={self.state.settings.model}")
            for i in range(1, 4):
                slot = self.state.slots[i-1]
                if slot.empty or slot.dropped:
                    await self._status(f"Skip slot {i}: empty={slot.empty} dropped={slot.dropped}")
                    continue
                await self._status(f"Running slot {i}…")
                await self._run_one(i)
            await self._status("Run complete.")

    async def action_add_new(self) -> None:
        # find first empty slot
        idx = next((i for i, s in enumerate(self.state.slots) if s.empty), None)
        if idx is None:
            await self._status("No empty slot. Drop one first with 'Xd'.")
            return
        self.run_worker(self._add_new_any_empty(), exclusive=True)

    async def action_edit_globals(self) -> None:
        # Simple inline JSON editor of globals
        data = {
            "model": self.state.settings.model,
            "temperature": self.state.settings.temperature,
            "seed": self.state.settings.seed,
            "num_ctx": self.state.settings.num_ctx,
            "num_predict": self.state.settings.num_predict,
            "top_p": self.state.settings.top_p,
            "top_k": self.state.settings.top_k,
            "repeat_penalty": self.state.settings.repeat_penalty,
            "presence_penalty": self.state.settings.presence_penalty,
            "frequency_penalty": self.state.settings.frequency_penalty,
            "base_url": self.state.settings.base_url,
        }
        raw = json.dumps(data, indent=2)
        edited = await self._prompt_multiline("Edit globals (JSON)", raw)
        if not edited:
            return
        try:
            obj = json.loads(edited)
            # Lock everything to user-provided/fixed values
            s = self.state.settings
            s.model = str(obj.get("model", s.model))
            s.temperature = float(obj.get("temperature", s.temperature))
            s.seed = int(obj.get("seed", s.seed))
            s.num_ctx = int(obj.get("num_ctx", s.num_ctx))
            s.num_predict = int(obj.get("num_predict", s.num_predict))
            s.top_p = float(obj.get("top_p", s.top_p))
            s.top_k = int(obj.get("top_k", s.top_k))
            s.repeat_penalty = float(obj.get("repeat_penalty", s.repeat_penalty))
            s.presence_penalty = float(obj.get("presence_penalty", s.presence_penalty))
            s.frequency_penalty = float(obj.get("frequency_penalty", s.frequency_penalty))
            s.base_url = str(obj.get("base_url", s.base_url))
            self.refresh_all()
        except Exception as e:
            await self._status(f"Invalid JSON: {e}")

    async def action_save_run(self) -> None:
        path = SAVE_DIR / f"run-{self.state.run_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state.to_json(), f, indent=2)
        await self._status(f"Saved: {path}")

    async def action_open_run(self) -> None:
        path = await self._prompt_line("Open run path (JSON)", str(SAVE_DIR))
        if not path:
            return
        p = Path(path).expanduser()
        if not p.exists():
            await self._status("File not found.")
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self.state = RunState.from_json(data)
            # Re-bind UI
            self.header_bar.state = self.state
            self.side_panel.state = self.state
            for i in range(3):
                self.panels[i+1].slot = self.state.slots[i]
            self.refresh_all()
        except Exception as e:
            await self._status(f"Open failed: {e}")

    async def action_quit(self) -> None:
        # Try to save; don’t block quitting if it fails.
        try:
            await self.action_save_run()
        except Exception as e:
            await self._status(f"Save failed: {e}")
        # Clean up background work & HTTP client, then exit the app.
        try:
            self.cancel_all_workers()
        except Exception:
            pass
        if getattr(self, "client", None) is not None:
            try:
                await self.client.aclose()
            except Exception:
                pass
        self.exit()

    # -------------- Helpers

    async def _add_new_in_slot(self, slot: int) -> None:
        s = self.state.slots[slot - 1]
        if not s.empty:
            await self._status(f"Slot {slot} not empty. Drop first with '{slot}d'.")
            return
        text = await self._prompt_multiline(f"New prompt for slot {slot}", "")
        if not text or not text.strip():
            return
        s.text = text.strip()
        s.empty = False
        s.active = True
        s.dropped = False
        s.born_stage = self.state.stage
        s.has_survived = False
        s.last_output = ""
        s.history.clear()
        self.panels[slot].slot = s
        self.panels[slot].refresh_prompt()
        self.side_panel.update_panel()

    async def _add_new_any_empty(self) -> None:
        idx = next((i for i, s in enumerate(self.state.slots) if s.empty), None)
        if idx is None:
            await self._status("No empty slot. Drop one first with 'Xd'.")
            return
        slot = idx + 1
        await self._add_new_in_slot(slot)

    async def _prompt_multiline(self, title: str, initial: str = "") -> Optional[str]:
        screen = PromptEditor(title, initial, multiline=True)
        # Must be called from a worker:
        return await self.push_screen_wait(screen)

    async def _prompt_line(self, title: str, initial: str = "") -> Optional[str]:
        screen = PromptEditor(title, initial, multiline=False)
        return await self.push_screen_wait(screen)

    async def _status(self, msg: str) -> None:
        """Write status to the side panel and to a log file."""
        # On-screen
        try:
            self.side_panel.post_status(msg)
        except Exception:
            pass
        # File log
        try:
            with (SAVE_DIR / "app.log").open("a", encoding="utf-8") as fh:
                fh.write(msg + "\n")
        except Exception:
            pass

    async def _edit_prompt(self, slot_id: int) -> None:
        slot = self.state.slots[slot_id-1]
        if slot.dropped:
            await self._status("This slot is dropped. Add or generate a new prompt.")
            return
        initial = "" if slot.empty else slot.text
        text = await self._prompt_multiline(f"Edit Prompt {slot_id}", initial)
        if text is None:
            return
        slot.text = text.strip()
        slot.empty = (slot.text.strip() == "")
        slot.active = not slot.empty
        self.panels[slot_id].slot = slot
        self.panels[slot_id].refresh_prompt()
        self.side_panel.update_panel()

    async def _drop_slot(self, slot_id: int) -> None:
        slot = self.state.slots[slot_id-1]
        if slot.empty or slot.dropped:
            await self._status("Slot already empty/dropped.")
            return
        # Perform drop: increment stage, compute scores, mark survivors as survived
        # Stage increments AFTER the drop. The dropped prompt final_score = 1 / stage.
        self.state.stage += 1
        slot.dropped = True
        slot.active = False
        slot.last_dropped_text = slot.text
        slot.final_score = 1.0 / self.state.stage if self.state.stage > 0 else 1.0
        # mark others as having survived at least one drop
        for s in self.state.slots:
            if s is not slot and not s.empty and not s.dropped:
                s.has_survived = True
        # Clear prompt content to free the slot but retain last_dropped_text for generator.
        slot.text = ""
        slot.empty = True
        slot.last_output = ""
        self.panels[slot_id].slot = slot
        self.panels[slot_id].refresh_prompt()
        self.panels[slot_id].clear_output()
        self.side_panel.update_panel()
        self.header_bar.refresh_bar()

    async def _generate_alternative(self, slot_id: int) -> None:
        slot = self.state.slots[slot_id-1]
        if not slot.empty:
            await self._status(f"Slot {slot} not empty. Drop first with '{slot}d'.")
            return
        base_text = slot.last_dropped_text or ""
        if not base_text.strip():
            await self._status(f"Slot {slot_id} has no dropped prompt to rewrite. Drop here first with '{slot_id}d'.")
            return
        try:
            await self._status(f"Generating alternative with {self.state.gen_settings.model} …")
            alt = await self._rewrite_prompt(base_text)
        except Exception as e:
            await self._status(f"Generate alternative failed: {e!s}")
            return
        # Insert new prompt
        slot.text = alt.strip()
        slot.empty = False
        slot.active = True
        slot.dropped = False
        slot.born_stage = self.state.stage
        slot.has_survived = False
        slot.last_output = ""
        slot.history.clear()
        self.panels[slot_id].slot = slot
        self.panels[slot_id].refresh_prompt()
        self.side_panel.update_panel()

    async def _run_one(self, slot_id: int) -> None:
        slot = self.state.slots[slot_id-1]
        panel = self.panels[slot_id]
        panel.clear_output()
        slot.last_output = ""
        try:
            async for chunk in self._ollama_stream(slot.text):
                if chunk is None:
                    continue
                panel.append_output(chunk)
                slot.last_output += chunk
            # end-of-stream: record stats are attached via last meta
        except Exception as e:
            panel.append_output(f"\n[ERROR] {e}")
        # store history
        rec = OutputRecord(
            timestamp=datetime.now(UTC).isoformat(),
            text=slot.last_output,
            stats=GenStats(),  # filled by _ollama_stream end meta if available
        )
        slot.history.append(rec)

    async def _ollama_stream(self, prompt: str):
        """Stream generation from Ollama with locked params; yields text chunks."""
        s = self.state.settings
        url = f"{s.base_url.rstrip('/')}/api/generate"
        await self._status(f"POST {url} model={s.model}")
        payload = {
            "model": s.model,
            "prompt": prompt,
            "stream": True,
            "seed": s.seed,
            "options": {
                "temperature": s.temperature,
                "top_p": s.top_p,
                "top_k": s.top_k,
                "repeat_penalty": s.repeat_penalty,
                "presence_penalty": s.presence_penalty,
                "frequency_penalty": s.frequency_penalty,
                "num_ctx": min(max(512, s.num_ctx), 131072),
                "num_predict": min(max(1, s.num_predict), 131072),
            },
        }
        try:
            async with self.client.stream("POST", url, json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        # Some servers may send "data: {...}" style; strip prefix
                        if line.startswith("data:"):
                            try:
                                obj = json.loads(line[5:].strip())
                            except Exception:
                                continue
                        else:
                            continue
                    if "response" in obj and obj.get("done") is not True:
                        yield obj["response"]
                    if obj.get("done"):
                        # Final stats could be used if desired
                        break
        except httpx.HTTPError as e:
            await self._status(f"Ollama HTTP error: {e!s}")
            raise
        except Exception as e:
            await self._status(f"Ollama stream error: {e!s}")
            raise

    async def _rewrite_prompt(self, dropped_text: str) -> str:
        """Use LlamaIndex (Ollama LLM) to propose a revised alternative of dropped prompt only."""
        try:
            from llama_index.llms.ollama import Ollama  # type: ignore
        except Exception:
            # Fallback: ask Ollama directly with a non-deterministic call
            return await self._rewrite_prompt_fallback(dropped_text)

        g = self.state.gen_settings
        llm = Ollama(
            model=g.model,
            base_url=self.state.settings.base_url,
            temperature=g.temperature,
            request_timeout=30.0,
            additional_kwargs={
                "top_p": g.top_p,
                "num_ctx": g.num_ctx,
                "num_predict": g.num_predict,
            },
        )
        system = (
            "You rewrite prompts for A/B testing of wording only. "
            "Rewrite the provided prompt into a clear, distinct alternative for the SAME task. "
            "Do not add new capabilities, tool calls, or constraints. Keep it roughly same length. "
            "Return ONLY the rewritten prompt text—no commentary."
        )
        user = f"Original prompt:\n---\n{dropped_text}\n---\nRewrite it."
        resp = await llm.acomplete(system_prompt=system, prompt=user)
        text = getattr(resp, "text", None) or str(resp)
        return text.strip()

    async def _rewrite_prompt_fallback(self, dropped_text: str) -> str:
        # Non-deterministic generator via direct Ollama; higher temp
        g = self.state.gen_settings
        s = self.state.settings
        url = f"{s.base_url.rstrip('/')}/api/generate"
        prompt = (
            "You rewrite prompts for A/B testing of wording only. "
            "Rewrite the provided prompt into a clear, distinct alternative for the SAME task. "
            "Do not add new capabilities, tool calls, or constraints. Keep it roughly same length.\n\n"
            f"Original prompt:\n---\n{dropped_text}\n---\nReturn ONLY the rewritten prompt."
        )
        payload = {
            "model": g.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": g.temperature,
                "top_p": g.top_p,
                "num_ctx": g.num_ctx,
                "num_predict": g.num_predict,
            },
        }
        try:
            resp = await self.client.post(url, json=payload, timeout=30.0)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            await self._status(f"Generator HTTP error: {e!s}")
            raise
        data = resp.json()
        return data.get("response", "").strip()

# -------------------------------
# Entrypoint
# -------------------------------

if __name__ == "__main__":
    try:
        app = PromptEvalApp()
        app.run()
    except KeyboardInterrupt:
        pass
