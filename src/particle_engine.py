import numpy as np
import cv2


# ─────────────────────────────────────────────
#  All particle / rendering constants
# ─────────────────────────────────────────────
MAX_SPARKS       = 60_000
SPARK_FRICTION   = 0.87
SPARK_GRAVITY    = 0.05
CANVAS_DECAY     = 0.78   # how fast old light fades (lower = longer trails)
BLOOM_STRENGTH   = 0.55   # additive bloom blend strength


class ParticleEngine:
    """
    Pure NumPy spark engine — only used for the explosive bursts
    launched from fast-moving wrists / ankles.
    """

    def __init__(self, W, H):
        self.W = W
        self.H = H
        n = MAX_SPARKS
        self.pos     = np.zeros((n, 2), dtype=np.float32)
        self.vel     = np.zeros((n, 2), dtype=np.float32)
        self.life    = np.zeros(n,      dtype=np.float32)
        self.maxlife = np.zeros(n,      dtype=np.float32)
        self.col     = np.zeros((n, 3), dtype=np.float32)
        self.size    = np.ones(n,       dtype=np.float32)
        self.n       = 0

    # ────────────────────────────────────────
    def burst(self, cx, cy, dx, dy, color, speed):
        """Radial burst, count & force scaled by landmark speed."""
        count = int(np.clip(speed * 8, 30, 350))
        n     = min(count, MAX_SPARKS - self.n)
        if n <= 0:
            return

        i, j = self.n, self.n + n

        self.pos[i:j, 0] = cx
        self.pos[i:j, 1] = cy

        # Emit in full 360° plus carry the limb's momentum
        angles  = np.random.uniform(0,   2*np.pi, n)
        radii   = np.random.uniform(0.2, 1.0,     n)
        force   = np.clip(speed * 0.4, 3, 18)

        self.vel[i:j, 0] = np.cos(angles) * radii * force + dx * 0.45
        self.vel[i:j, 1] = np.sin(angles) * radii * force + dy * 0.45

        base_life = np.random.uniform(30, 75, n)
        self.life[i:j]    = base_life
        self.maxlife[i:j] = base_life

        # Colour variation ± 30 %
        var = np.random.uniform(0.7, 1.3, (n, 3))
        self.col[i:j] = np.clip(np.array(color, dtype=np.float32) * var, 0, 255)

        # Small sparks are smaller, fast sparks are bigger
        self.size[i:j] = np.random.uniform(0.5, 2.5, n)

        self.n = j

    # ────────────────────────────────────────
    def update(self):
        if self.n == 0:
            return
        n = self.n
        self.pos[:n]    += self.vel[:n]
        self.vel[:n]    *= SPARK_FRICTION
        self.vel[:n, 1] += SPARK_GRAVITY
        self.life[:n]   -= 1

        alive   = self.life[:n] > 0
        na      = int(alive.sum())
        if na < n:
            for arr in (self.pos, self.vel, self.life, self.maxlife, self.col, self.size):
                arr[:na] = arr[:n][alive]
        self.n = na

    # ────────────────────────────────────────
    def render_to(self, canvas: np.ndarray):
        """Additively paint sparks onto the float32 canvas."""
        if self.n == 0:
            return
        n = self.n
        c = self.pos[:n].astype(np.int32)
        valid = (
            (c[:, 0] >= 1) & (c[:, 0] < self.W - 2) &
            (c[:, 1] >= 1) & (c[:, 1] < self.H - 2)
        )
        cx = c[valid, 0]
        cy = c[valid, 1]

        ratio = (self.life[:n][valid] / np.maximum(self.maxlife[:n][valid], 1))
        col   = self.col[:n][valid] * ratio.reshape(-1, 1)

        # Use np.add.at so overlapping sparks stack correctly
        np.add.at(canvas, (cy,   cx),   col)
        np.add.at(canvas, (cy-1, cx),   col * 0.35)
        np.add.at(canvas, (cy+1, cx),   col * 0.35)
        np.add.at(canvas, (cy,   cx-1), col * 0.35)
        np.add.at(canvas, (cy,   cx+1), col * 0.35)
