"""
renderer.py  — Cinematic multi-layer glow renderer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each frame is composed of 5 layers:
  1.  Dense stardust point-cloud along every bone
  2.  Hot neon core lines (thin, very bright)
  3.  Soft bloom (Gaussian blur of the canvas blended back)
  4.  Explosive spark physics from fast extremities
  5.  Long sweeping motion-trail arcs
"""

import cv2
import numpy as np
from collections import deque

# ─── Color palette (BGR float32) ──────────────────────────────────────────────
C_CYAN    = np.array([255, 240,  60], dtype=np.float32)
C_MAGENTA = np.array([210,  40, 255], dtype=np.float32)
C_GOLD    = np.array([ 30, 195, 255], dtype=np.float32)
C_WHITE   = np.array([255, 255, 255], dtype=np.float32)
C_BLUE    = np.array([255, 100,  60], dtype=np.float32)

# ─── Skeleton connection table ─────────────────────────────────────────────────
CONNECTIONS = [
    # torso
    (11, 12, C_MAGENTA), (11, 23, C_MAGENTA),
    (12, 24, C_MAGENTA), (23, 24, C_MAGENTA),
    # left arm
    (11, 13, C_CYAN), (13, 15, C_CYAN),
    (15, 17, C_CYAN), (15, 19, C_CYAN), (15, 21, C_CYAN),
    # right arm
    (12, 14, C_MAGENTA), (14, 16, C_MAGENTA),
    (16, 18, C_MAGENTA), (16, 20, C_MAGENTA), (16, 22, C_MAGENTA),
    # left leg
    (23, 25, C_GOLD), (25, 27, C_GOLD), (27, 29, C_GOLD), (27, 31, C_GOLD),
    # right leg
    (24, 26, C_GOLD), (26, 28, C_GOLD), (28, 30, C_GOLD), (28, 32, C_GOLD),
    # face cross
    (0,  4,  C_BLUE), (0,  1,  C_BLUE),
]

EXTREMITIES = {
    15: C_CYAN,    # left wrist
    16: C_MAGENTA, # right wrist
    27: C_GOLD,    # left ankle
    28: C_GOLD,    # right ankle
}


class Renderer:

    def __init__(self, W, H):
        self.W = W
        self.H = H
        # Persistent accumulation buffer (float32 for precision)
        self.canvas = np.zeros((H, W, 3), dtype=np.float32)
        # Trail deques
        self.trails: dict[int, deque] = {k: deque(maxlen=45) for k in EXTREMITIES}

    # ──────────────────────────────────────────────────────────────────────────
    def _add_stardust_segment(self, p1, p2, color, n_pts=90, scatter=9):
        """
        Paint a dense probabilistic cloud of glowing dots along one bone.
        Each dot is a small soft circle for a genuine stardust look.
        """
        ts  = np.random.uniform(0, 1, n_pts)
        pts = (p1 + np.outer(ts, p2 - p1)
               + np.random.normal(0, scatter, (n_pts, 2)))
        pts[:, 0] = np.clip(pts[:, 0], 2, self.W - 3)
        pts[:, 1] = np.clip(pts[:, 1], 2, self.H - 3)

        brightness = np.random.uniform(0.25, 1.0, n_pts)
        for i in range(n_pts):
            b  = brightness[i]
            c  = (color * b).astype(np.float32)
            x, y = int(pts[i, 0]), int(pts[i, 1])
            # 3×3 soft dot
            self.canvas[y-1:y+2, x-1:x+2] += c * 0.4
            # single bright centre
            self.canvas[y, x] += c

    # ──────────────────────────────────────────────────────────────────────────
    def _draw_neon_core(self, p1, p2, color):
        """
        Multi-pass line drawing to simulate a neon tube:
          pass 1 — very thick, dim outer glow
          pass 2 — medium, mid-bright
          pass 3 — thin bright core
          pass 4 — 1-px pure white hot core
        """
        tmp = np.zeros_like(self.canvas)
        i1 = (int(p1[0]), int(p1[1]))
        i2 = (int(p2[0]), int(p2[1]))

        cv2.line(tmp, i1, i2, (color * 0.25).tolist(), 14, cv2.LINE_AA)
        cv2.line(tmp, i1, i2, (color * 0.55).tolist(),  7, cv2.LINE_AA)
        cv2.line(tmp, i1, i2, (color * 0.90).tolist(),  3, cv2.LINE_AA)
        cv2.line(tmp, i1, i2, C_WHITE.tolist(),          1, cv2.LINE_AA)

        self.canvas += tmp

    # ──────────────────────────────────────────────────────────────────────────
    def _draw_joint_glow(self, x, y, color, radius=7):
        """Bright pulsing dot at each joint."""
        tmp = np.zeros_like(self.canvas)
        cv2.circle(tmp, (x, y), radius + 6, (color * 0.25).tolist(), -1, cv2.LINE_AA)
        cv2.circle(tmp, (x, y), radius + 2, (color * 0.65).tolist(), -1, cv2.LINE_AA)
        cv2.circle(tmp, (x, y), radius,      (color        ).tolist(), -1, cv2.LINE_AA)
        cv2.circle(tmp, (x, y), max(1, radius - 2), C_WHITE.tolist(), -1, cv2.LINE_AA)
        self.canvas += tmp

    # ──────────────────────────────────────────────────────────────────────────
    def _bloom_pass(self):
        """
        Blur the current canvas and blend it back additively.
        This creates the iconic soft halo / bloom around all lights.
        """
        c8 = np.clip(self.canvas, 0, 255).astype(np.uint8)
        # Two blur scales — wide halo + medium softening
        blur_big  = cv2.GaussianBlur(c8, (55, 55), 0).astype(np.float32)
        blur_med  = cv2.GaussianBlur(c8, (21, 21), 0).astype(np.float32)
        self.canvas += blur_big * 0.20 + blur_med * 0.15

    # ──────────────────────────────────────────────────────────────────────────
    def _draw_trails(self, display: np.ndarray):
        """
        Draw the sweeping neon arc trails for extremities directly on the
        final display (NOT the persistent canvas) so they feel immediate.
        """
        for idx, color in EXTREMITIES.items():
            trail = self.trails.get(idx)
            if not trail or len(trail) < 2:
                continue
            pts = list(trail)
            n   = len(pts)
            for i in range(1, n):
                a = i / n
                # Outer faint glow
                gc = (color * a * 0.35).astype(np.uint8).tolist()
                cv2.line(display, pts[i-1], pts[i], gc, int(a * 9) + 3, cv2.LINE_AA)
                # Mid layer
                mc = (color * a * 0.75).astype(np.uint8).tolist()
                cv2.line(display, pts[i-1], pts[i], mc, int(a * 5) + 1, cv2.LINE_AA)
                # Bright white core
                wc = np.clip(color * a + C_WHITE * (a * 0.4), 0, 255).astype(np.uint8).tolist()
                cv2.line(display, pts[i-1], pts[i], wc, max(1, int(a * 2)), cv2.LINE_AA)

    # ──────────────────────────────────────────────────────────────────────────
    def frame(
        self,
        lm_dict: dict,
        engine,               # ParticleEngine ref
        bg_frame=None
    ) -> np.ndarray:
        """
        Compose one full cinematic frame.

        Args:
            lm_dict  : {landmark_idx: (x, y)} from kinetic_ghost
            engine   : ParticleEngine instance
            bg_frame : Optional darkened webcam frame (for bg_mode=1)
        Returns:
            Final BGR uint8 display frame
        """

        # ── 1. Decay the persistent canvas ──────────────────────────────────
        self.canvas *= 0.80
        np.clip(self.canvas, 0, 255, out=self.canvas)

        # ── 2. Body aura & neon skeleton ────────────────────────────────────
        if lm_dict:
            for (s, e, color) in CONNECTIONS:
                if s not in lm_dict or e not in lm_dict:
                    continue
                p1 = np.array(lm_dict[s], dtype=np.float32)
                p2 = np.array(lm_dict[e], dtype=np.float32)

                # a) dense stardust cloud along the bone
                self._add_stardust_segment(p1, p2, color, n_pts=100, scatter=8)

                # b) glowing neon tube at the core
                self._draw_neon_core(p1, p2, color)

            # Glowing dots at every visible joint
            joint_colors = {}
            for (s, e, color) in CONNECTIONS:
                joint_colors[s] = color
                joint_colors[e] = color
            for idx, (x, y) in lm_dict.items():
                c = joint_colors.get(idx, C_WHITE)
                self._draw_joint_glow(x, y, c, radius=6)

        # ── 3. Bloom pass (expensive but cinematic) ──────────────────────────
        self._bloom_pass()

        # ── 4. Clamp & convert to uint8 for display ──────────────────────────
        canvas_u8 = np.clip(self.canvas, 0, 255).astype(np.uint8)

        # ── 5. Compose over background ────────────────────────────────────────
        if bg_frame is not None:
            display = cv2.addWeighted(bg_frame, 1.0, canvas_u8, 1.0, 0)
        else:
            display = canvas_u8.copy()

        # ── 6. Render explosion sparks on top ────────────────────────────────
        engine.render_to(
            display.astype(np.float32)
        )
        # (render_to writes to a float canvas; we need to merge)
        spark_canvas = np.zeros_like(self.canvas)
        engine.render_to(spark_canvas)
        display = np.clip(
            display.astype(np.float32) + spark_canvas, 0, 255
        ).astype(np.uint8)

        # ── 7. Sweeping neon arc trails (drawn directly) ──────────────────────
        self._draw_trails(display)

        # ── 8. Final subtle vignette for cinematic feel ───────────────────────
        display = self._vignette(display)

        return display

    # ──────────────────────────────────────────────────────────────────────────
    def _vignette(self, img: np.ndarray) -> np.ndarray:
        """Dark circular vignette around edges for that cinematic look."""
        H, W = img.shape[:2]
        cx, cy = W / 2, H / 2
        Y, X   = np.ogrid[:H, :W]
        dist   = np.sqrt((X - cx)**2 + (Y - cy)**2) / np.sqrt(cx**2 + cy**2)
        mask   = np.clip(1.0 - dist * 0.65, 0, 1).astype(np.float32)
        mask3  = mask[:, :, np.newaxis]
        return (img.astype(np.float32) * mask3).astype(np.uint8)
