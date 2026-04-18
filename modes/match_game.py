# modes/match_game.py — Minijuego: conectar herramientas de pago con sus alternativas OSS
#
# Mecánica:
#   · En el lado IZQUIERDO aparecen íconos de herramientas de pago
#   · En el lado DERECHO aparecen sus equivalentes open source
#   · El jugador hace pinch (pulgar+índice) para "agarrar" un ícono del lado izquierdo
#   · Arrastra con la mano hasta el ícono correspondiente del lado derecho
#   · Al soltar (abrir la mano), se verifica si la conexión es correcta
#   · La UI (paneles de íconos) NO interfiere con la cámara: se superpone semitransparente
#
# Estructura de un par:
#   {
#       "paid":     {"name": "Photoshop",   "file": "photoshop.png"},
#       "oss":      {"name": "Photopea",    "file": "photopea.png"},
#   }
#
# Assets esperados:
#   assets/icons_paid/  → íconos de herramientas de pago (PNG con alpha, 80x80 ideal)
#   assets/icons_oss/   → íconos de alternativas OSS (mismo nombre o mapeo en PAIRS)

import os
import time
import random
import cv2
import numpy as np

import config
from core.overlay import (
    overlay_image, draw_panel, draw_text, draw_text_centered,
    draw_score_hud, draw_controls_help, draw_pinch_cursor,
    draw_connection_line, load_asset, placeholder_asset
)


# ── Definición de pares ───────────────────────────────────────────────────────

PAIRS = [
    {"paid": {"name": "Photoshop",  "file": "photoshop.png"},  "oss": {"name": "Photopea",   "file": "photopea.png"}},
    {"paid": {"name": "Slack",      "file": "slack.png"},       "oss": {"name": "Mattermost", "file": "mattermost.png"}},
    {"paid": {"name": "GitHub",     "file": "github.png"},      "oss": {"name": "Gitea",      "file": "gitea.png"}},
    {"paid": {"name": "Notion",     "file": "notion.png"},      "oss": {"name": "Obsidian",   "file": "obsidian.png"}},
    {"paid": {"name": "Figma",      "file": "figma.png"},       "oss": {"name": "Penpot",     "file": "penpot.png"}},
    {"paid": {"name": "Spotify",    "file": "spotify.png"},     "oss": {"name": "Funkwhale",  "file": "funkwhale.png"}},
    {"paid": {"name": "Windows",    "file": "windows.png"},     "oss": {"name": "Linux",      "file": "linux.png"}},
]

_ICON_SIZE   = 80    # Tamaño de cada ícono en píxeles
_ICON_PAD    = 20    # Espacio entre íconos en la columna
_COL_WIDTH   = 150   # Ancho de cada columna de íconos


class MatchGame:
    """
    Gestiona el minijuego completo de matching.
    Se instancia una vez y se llama a update() en cada frame.
    """

    CONTROLS = {
        "Pinch": "Agarrar y arrastrar",
        "Soltar": "Conectar",
        "R":     "Reiniciar ronda",
        "1/2":   "Cambiar modo",
        "Q":     "Salir",
    }

    def __init__(self):
        self._icons_paid = {}   # name → np.ndarray BGRA
        self._icons_oss  = {}   # name → np.ndarray BGRA
        self._load_icons()

        self._score        = 0
        self._matches_done = 0
        self._round_pairs  = []         # Subconjunto de PAIRS para la ronda actual
        self._slots_paid   = []         # Lista de SlotData lado izquierdo
        self._slots_oss    = []         # Lista de SlotData lado derecho
        self._connections  = []         # Conexiones correctas ya dibujadas
        self._wrong_flash  = 0.0        # Timestamp hasta el que mostrar flash de error

        # Estado del arrastre activo
        self._dragging_paid_idx = None  # Índice del slot paid que se está arrastrando
        self._drag_start_pt     = None  # Punto donde se inició el arrastre (px)
        self._drag_current_pt   = None  # Punto actual de la mano

        self._reset_round()

    # ── API pública ───────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray, hand_results) -> np.ndarray:
        """
        Procesa el estado del juego y dibuja el frame de salida.

        Args:
            frame        : Frame BGR de la cámara (no se modifica el original).
            hand_results : Objeto HandResults de hand_tracker.py

        Returns:
            np.ndarray — frame con toda la UI del juego superpuesta.
        """
        display = frame.copy()

        # Actualizar estado de arrastre con la primera mano disponible
        hand = hand_results.first
        self._update_drag_state(hand)

        # Dibujar columnas de íconos
        h, w = frame.shape[:2]
        self._draw_paid_column(display, w, h)
        self._draw_oss_column(display, w, h)

        # Dibujar conexiones correctas ya establecidas
        for conn in self._connections:
            draw_connection_line(display, conn["p_paid"], conn["p_oss"],
                                 color=config.COLOR_GREEN)

        # Dibujar línea activa de arrastre
        if self._dragging_paid_idx is not None and self._drag_current_pt:
            start = self._get_slot_center("paid", self._dragging_paid_idx)
            draw_connection_line(display, start, self._drag_current_pt,
                                 color=config.LINE_COLOR)

        # Cursor de la mano
        if hand:
            draw_pinch_cursor(display, hand.pinch_midpoint, hand.is_pinching)

        # Flash de error
        if time.time() < self._wrong_flash:
            self._draw_error_flash(display)

        # HUD de puntuación
        draw_score_hud(display, self._score, self._matches_done, len(self._round_pairs))

        # Pantalla de victoria si se completaron todos los pares
        if self._matches_done >= len(self._round_pairs):
            self._draw_win_screen(display)

        # Controles
        draw_controls_help(display, self.CONTROLS)

        return display

    def handle_key(self, key: int) -> str | None:
        if key == ord('r') or key == ord('R'):
            self._reset_round()
        elif key == config.KEY_MODE_BOOTH:
            return "mode_booth"
        elif key == config.KEY_QUIT:
            return "quit"
        return None

    def close(self):
        print("[MatchGame] Recursos liberados.")

    # ── Lógica de arrastre ────────────────────────────────────────────────────

    def _update_drag_state(self, hand):
        """
        Máquina de estados del arrastre:
            IDLE      → si hay pinch sobre un ícono paid → DRAGGING
            DRAGGING  → si se suelta → verificar conexión → IDLE
        """
        if hand is None:
            # Sin mano: cancela arrastre
            self._drag_current_pt   = None
            self._dragging_paid_idx = None
            return

        cursor = hand.pinch_midpoint
        self._drag_current_pt = cursor

        if hand.is_pinching:
            if self._dragging_paid_idx is None:
                # Intentar iniciar arrastre desde un slot paid no conectado
                idx = self._hit_test_paid(cursor)
                if idx is not None:
                    # Solo permite arrastrar si este slot no tiene conexión aún
                    already = any(c["paid_idx"] == idx for c in self._connections)
                    if not already:
                        self._dragging_paid_idx = idx
                        self._drag_start_pt     = cursor
        else:
            # Se soltó: verificar si aterrizó sobre un ícono oss
            if self._dragging_paid_idx is not None:
                oss_idx = self._hit_test_oss(cursor)
                if oss_idx is not None:
                    self._try_connect(self._dragging_paid_idx, oss_idx)
                self._dragging_paid_idx = None
                self._drag_start_pt     = None

    def _hit_test_paid(self, point) -> int | None:
        """Devuelve el índice del slot paid bajo el cursor, o None."""
        for i, slot in enumerate(self._slots_paid):
            if _in_rect(point, slot["rect"]):
                return i
        return None

    def _hit_test_oss(self, point) -> int | None:
        """Devuelve el índice del slot oss bajo el cursor, o None."""
        for i, slot in enumerate(self._slots_oss):
            if _in_rect(point, slot["rect"]) or _near(point, self._get_slot_center("oss", i)):
                return i
        return None

    def _try_connect(self, paid_idx: int, oss_idx: int):
        """
        Verifica si el par (paid_idx, oss_idx) es correcto.
        Si sí: agrega conexión correcta y suma puntos.
        Si no: activa flash de error.
        """
        paid_name = self._round_pairs[paid_idx]["paid"]["name"]
        oss_name  = self._round_pairs[paid_idx]["oss"]["name"]    # El correcto para este paid
        selected_oss_name = self._round_pairs[oss_idx]["oss"]["name"]

        if oss_name == selected_oss_name:
            # ¡Correcto!
            self._connections.append({
                "paid_idx": paid_idx,
                "oss_idx":  oss_idx,
                "p_paid":   self._get_slot_center("paid", paid_idx),
                "p_oss":    self._get_slot_center("oss",  oss_idx),
            })
            self._matches_done += 1
            self._score        += config.SCORE_PER_MATCH
            print(f"[MatchGame] ¡Correcto! {paid_name} → {oss_name}. Score: {self._score}")
        else:
            # Incorrecto
            self._wrong_flash = time.time() + 0.6    # Flash de 600ms
            self._score = max(0, self._score - 20)   # Pequeña penalización
            print(f"[MatchGame] Incorrecto: {paid_name} conectado con {selected_oss_name}")

    # ── Dibujo de columnas ────────────────────────────────────────────────────

    def _draw_paid_column(self, frame, w, h):
        """Dibuja la columna izquierda: herramientas de pago."""
        col_x = 10
        self._draw_column(frame, col_x, h, self._slots_paid, "De pago",
                          color=config.COLOR_RED, side="paid")

    def _draw_oss_column(self, frame, w, h):
        """Dibuja la columna derecha: alternativas OSS."""
        col_x = w - _COL_WIDTH - 10
        self._draw_column(frame, col_x, h, self._slots_oss, "Open Source",
                          color=config.COLOR_GREEN, side="oss")

    def _draw_column(self, frame, col_x, frame_h, slots, title, color, side):
        """Dibuja una columna de íconos con su título."""
        n     = len(slots)
        total = n * _ICON_SIZE + (n - 1) * _ICON_PAD + 50   # +50 para el título
        y0    = (frame_h - total) // 2

        # Panel de fondo
        draw_panel(frame, col_x - 10, y0 - 10, _COL_WIDTH + 20, total + 20,
                   alpha=0.55, radius=14)

        # Título de columna
        draw_text_centered(frame, title,
                           col_x + _COL_WIDTH // 2, y0 + 18,
                           color=color, scale=0.6, thickness=1)

        for i, slot in enumerate(slots):
            y = y0 + 40 + i * (_ICON_SIZE + _ICON_PAD)
            x = col_x

            # Guardar rect del slot para hit testing
            slot["rect"] = (x, y, x + _ICON_SIZE, y + _ICON_SIZE)

            # Determinar si este slot ya está conectado
            connected_idx = next(
                (c["oss_idx" if side == "paid" else "paid_idx"]
                 for c in self._connections
                 if c[f"{side}_idx"] == i),
                None
            )
            connected = connected_idx is not None

            # Fondo del slot (verde si conectado, neutro si no)
            bg_color = (0, 80, 0) if connected else (40, 40, 40)
            draw_panel(frame, x, y, _ICON_SIZE, _ICON_SIZE,
                       color_bgr=bg_color, alpha=0.7, radius=10)

            # Ícono
            icon = slot.get("icon")
            if icon is not None:
                frame = overlay_image(frame, icon, x, y, _ICON_SIZE, _ICON_SIZE)
            else:
                # Placeholder de texto si no hay imagen
                draw_text_centered(frame, slot["name"][:3],
                                   x + _ICON_SIZE // 2, y + _ICON_SIZE // 2,
                                   color=(200, 200, 200), scale=0.55, thickness=1)

            # Nombre debajo del ícono
            draw_text_centered(frame, slot["name"],
                               x + _ICON_SIZE // 2, y + _ICON_SIZE + 14,
                               color=color if connected else config.COLOR_WHITE,
                               scale=0.45, thickness=1)

            # Resaltado si se está arrastrando DESDE este slot
            if side == "paid" and i == self._dragging_paid_idx:
                cv2.rectangle(frame, (x - 3, y - 3),
                              (x + _ICON_SIZE + 3, y + _ICON_SIZE + 3),
                              config.COLOR_ACCENT, 3)

    # ── Pantallas especiales ──────────────────────────────────────────────────

    def _draw_error_flash(self, frame):
        """Overlay rojo semitransparente en pantalla al equivocarse."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]),
                      (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        draw_text_centered(frame, "Intenta de nuevo",
                           frame.shape[1] // 2, frame.shape[0] // 2,
                           color=config.COLOR_WHITE, scale=1.2,
                           thickness=2, bg_color=config.COLOR_UI_BG)

    def _draw_win_screen(self, frame):
        """Overlay de victoria cuando se conectan todos los pares."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 60, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        draw_text_centered(frame, "¡Completado!", w // 2, h // 2 - 40,
                           color=config.COLOR_GREEN, scale=2.0, thickness=4)
        draw_text_centered(frame, f"Puntuacion: {self._score}",
                           w // 2, h // 2 + 20,
                           color=config.COLOR_ACCENT, scale=1.2, thickness=2)
        draw_text_centered(frame, "Presiona R para nueva ronda",
                           w // 2, h // 2 + 70,
                           color=config.COLOR_WHITE, scale=0.8, thickness=1)

    # ── Gestión de ronda ──────────────────────────────────────────────────────

    def _reset_round(self):
        """Inicia una nueva ronda seleccionando pares aleatorios."""
        n = min(config.TOTAL_PAIRS, len(PAIRS))
        self._round_pairs   = random.sample(PAIRS, n)
        self._connections   = []
        self._matches_done  = 0
        self._score         = 0
        self._dragging_paid_idx = None

        # Construir slots
        self._slots_paid = [
            {"name": p["paid"]["name"], "icon": self._icons_paid.get(p["paid"]["name"]), "rect": None}
            for p in self._round_pairs
        ]
        self._slots_oss = [
            {"name": p["oss"]["name"],  "icon": self._icons_oss.get(p["oss"]["name"]),   "rect": None}
            for p in self._round_pairs
        ]
        # Barajar los OSS para que no estén alineados directamente frente al paid
        random.shuffle(self._slots_oss)
        print(f"[MatchGame] Nueva ronda con {n} pares.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_slot_center(self, side: str, idx: int):
        """Devuelve el centro (x, y) de un slot."""
        slots = self._slots_paid if side == "paid" else self._slots_oss
        rect  = slots[idx].get("rect")
        if rect is None:
            return (0, 0)
        x1, y1, x2, y2 = rect
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    # ── Carga de assets ───────────────────────────────────────────────────────

    def _load_icons(self):
        """
        Carga los íconos de ambas carpetas.
        Mapea el nombre del archivo al nombre del par para recuperarlos rápido.
        """
        for pair in PAIRS:
            # Ícono de pago
            paid_name = pair["paid"]["name"]
            paid_file = pair["paid"]["file"]
            paid_path = os.path.join("assets", "icons_paid", paid_file)
            img = load_asset(paid_path)
            self._icons_paid[paid_name] = img if img is not None else \
                placeholder_asset(_ICON_SIZE, _ICON_SIZE, (30, 30, 120), paid_name[:3])

            # Ícono OSS
            oss_name = pair["oss"]["name"]
            oss_file = pair["oss"]["file"]
            oss_path = os.path.join("assets", "icons_oss", oss_file)
            img = load_asset(oss_path)
            self._icons_oss[oss_name] = img if img is not None else \
                placeholder_asset(_ICON_SIZE, _ICON_SIZE, (0, 80, 0), oss_name[:3])

        paid_ok = sum(1 for v in self._icons_paid.values() if v is not None)
        oss_ok  = sum(1 for v in self._icons_oss.values()  if v is not None)
        print(f"[MatchGame] Íconos cargados — paid: {paid_ok}, oss: {oss_ok}")


# ── Utilidades ────────────────────────────────────────────────────────────────

def _in_rect(point, rect) -> bool:
    """True si el punto está dentro del rectángulo (x1,y1,x2,y2)."""
    if rect is None:
        return False
    px, py = point
    x1, y1, x2, y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2


def _near(p1, p2, radius: int = config.MATCH_RADIUS) -> bool:
    """True si dos puntos están dentro de un radio dado."""
    import math
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1]) < radius
