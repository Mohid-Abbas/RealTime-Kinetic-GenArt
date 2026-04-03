"""
kinetic_ghost.py  — Pose pipeline + main loop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Responsibilities:
  • Capture + flip webcam frames
  • Run MediaPipe Holistic
  • Smooth landmark jitter
  • Compute per-landmark velocity
  • Route data to Renderer and ParticleEngine
  • Handle keybinds + window
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque

from src.particle_engine import ParticleEngine
from src.renderer        import Renderer, EXTREMITIES, CONNECTIONS


# ─── Moving-average smoother ─────────────────────────────────────────────────
class Smoother:
    def __init__(self, window=6):
        self._h: dict[int, deque] = {}
        self._w = window

    def __call__(self, idx: int, x: int, y: int):
        if idx not in self._h:
            self._h[idx] = deque(maxlen=self._w)
        self._h[idx].append((x, y))
        xs = [p[0] for p in self._h[idx]]
        ys = [p[1] for p in self._h[idx]]
        return int(sum(xs)/len(xs)), int(sum(ys)/len(ys))


# ─── Main Application ─────────────────────────────────────────────────────────
class KineticGhostApp:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.W = W if W > 0 else 1280
        self.H = H if H > 0 else 720

        self.smoother = Smoother(window=6)
        self.engine   = ParticleEngine(self.W, self.H)
        self.renderer = Renderer(self.W, self.H)

        mp_h = mp.solutions.holistic
        self.holistic = mp_h.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2          # highest accuracy model
        )

        self.prev_lm: dict[int, tuple] = {}
        self.bg_mode = 0    # 0 = black,  1 = darkened webcam

    # ──────────────────────────────────────────────────────────────────────────
    def _extract_landmarks(self, results) -> dict:
        """Return {idx: (sx, sy)} for all visible pose landmarks."""
        out = {}
        if not results.pose_landmarks:
            return out
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            if lm.visibility > 0.45:
                rx = int(lm.x * self.W)
                ry = int(lm.y * self.H)
                sx, sy = self.smoother(idx, rx, ry)
                out[idx] = (sx, sy)
        return out

    # ──────────────────────────────────────────────────────────────────────────
    def _emit_extremity_bursts(self, lm_dict: dict):
        """
        For each fast-moving extremity, push a burst of physics sparks
        into the particle engine.
        """
        for idx, color in EXTREMITIES.items():
            if idx not in lm_dict:
                continue
            sx, sy = lm_dict[idx]

            # Update trail
            self.renderer.trails[idx].append((sx, sy))

            # Velocity
            if idx in self.prev_lm:
                px, py = self.prev_lm[idx]
                dx, dy = sx - px, sy - py
                speed  = np.sqrt(dx**2 + dy**2)
                if speed > 3:
                    self.engine.burst(sx, sy, dx, dy, color.tolist(), speed)

            self.prev_lm[idx] = (sx, sy)

    # ──────────────────────────────────────────────────────────────────────────
    def run(self):
        print("═══════════════════════════════════════")
        print("  KineticGhost.ai  |  Cinematic Mode  ")
        print("  b = toggle bg    |  q = quit        ")
        print("═══════════════════════════════════════")

        cv2.namedWindow('KineticGhost', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('KineticGhost', self.W, self.H)

        while self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)

            # MediaPipe needs RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self.holistic.process(rgb)

            lm_dict = self._extract_landmarks(results)
            self._emit_extremity_bursts(lm_dict)

            # Physics tick
            self.engine.update()

            # Background layer
            bg = None
            if self.bg_mode == 1:
                bg = cv2.convertScaleAbs(frame, alpha=0.12, beta=0)

            # Compose cinematic frame
            display = self.renderer.frame(lm_dict, self.engine, bg_frame=bg)

            cv2.imshow('KineticGhost', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                self.bg_mode ^= 1

        self.cap.release()
        cv2.destroyAllWindows()
