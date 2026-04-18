# Renderizado de UI e imágenes sobre frames de OpenCV
#
# Responsabilidades:
#   · Superponer imágenes PNG con canal alpha (stickers, íconos)
#   · Dibujar paneles semitransparentes de UI
#   · Renderizar texto con fondo para legibilidad
#   · Dibujar elementos del HUD: botones, barras, contadores
#
# IMPORTANTE: Todas las funciones reciben y devuelven frames BGR de numpy.
#             Los stickers/íconos deben ser arrays BGRA (con canal alpha).

import cv2
import numpy as np
import config


# ── Composición de imágenes ───────────────────────────────────────────────────

def overlay_image(background: np.ndarray,
                  overlay_bgra: np.ndarray,
                  x: int, y: int,
                  width: int = None, height: int = None) -> np.ndarray:
    """
    Superpone una imagen BGRA sobre un frame BGR usando el canal alpha.

    Args:
        background   : Frame BGR de destino (se modifica en lugar).
        overlay_bgra : Imagen PNG cargada con cv2.IMREAD_UNCHANGED (4 canales).
        x, y         : Esquina superior-izquierda donde colocar el sticker.
        width, height: Si se especifican, redimensiona el overlay antes de pegar.

    Returns:
        El frame background modificado (misma referencia, útil para encadenar).
    """
    if overlay_bgra is None:
        return background

    # Redimensionar si se pidió
    if width is not None and height is not None:
        overlay_bgra = cv2.resize(overlay_bgra, (width, height),
                                  interpolation=cv2.INTER_AREA)

    ov_h, ov_w = overlay_bgra.shape[:2]
    bg_h, bg_w = background.shape[:2]

    # Recortar si el sticker se sale del frame
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + ov_w, bg_w), min(y + ov_h, bg_h)
    if x2 <= x1 or y2 <= y1:
        return background

    # Región del overlay que cae dentro del frame
    ov_x1 = x1 - x
    ov_y1 = y1 - y
    ov_x2 = ov_x1 + (x2 - x1)
    ov_y2 = ov_y1 + (y2 - y1)

    # Separar canales
    overlay_crop = overlay_bgra[ov_y1:ov_y2, ov_x1:ov_x2]
    b, g, r, a   = cv2.split(overlay_crop)
    alpha_f      = a.astype(np.float32) / 255.0   # Máscara [0.0, 1.0]
    alpha_inv    = 1.0 - alpha_f

    roi = background[y1:y2, x1:x2].astype(np.float32)
    fg  = np.stack([b, g, r], axis=2).astype(np.float32)

    # Alpha blending: resultado = fg * alpha + bg * (1 - alpha)
    for c in range(3):
        roi[:, :, c] = fg[:, :, c] * alpha_f + roi[:, :, c] * alpha_inv

    background[y1:y2, x1:x2] = roi.astype(np.uint8)
    return background


def overlay_image_centered(background, overlay_bgra, cx, cy, width, height):
    """
    Versión centrada de overlay_image: cx, cy son el centro del sticker.
    Útil para anclar stickers a landmarks (e.g., centro de la cabeza).
    """
    x = cx - width  // 2
    y = cy - height // 2
    return overlay_image(background, overlay_bgra, x, y, width, height)


# ── Paneles semitransparentes ─────────────────────────────────────────────────

def draw_panel(frame: np.ndarray,
               x: int, y: int, w: int, h: int,
               color_bgr: tuple = config.COLOR_UI_BG,
               alpha: float = config.UI_ALPHA,
               radius: int = 12) -> np.ndarray:
    """
    Dibuja un rectángulo semitransparente con esquinas redondeadas.
    Útil para fondos de menús, HUD, contadores.
    """
    overlay = frame.copy()

    # Rectángulo principal
    cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), color_bgr, -1)
    cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), color_bgr, -1)

    # Cuatro esquinas redondeadas
    corners = [
        (x + radius,         y + radius),
        (x + w - radius,     y + radius),
        (x + radius,         y + h - radius),
        (x + w - radius,     y + h - radius),
    ]
    for cx, cy in corners:
        cv2.circle(overlay, (cx, cy), radius, color_bgr, -1)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


# ── Texto con fondo ───────────────────────────────────────────────────────────

def draw_text(frame: np.ndarray,
              text: str,
              x: int, y: int,
              color: tuple = config.COLOR_WHITE,
              scale: float = config.FONT_SCALE,
              thickness: int = config.FONT_THICKNESS,
              bg_color: tuple = None,
              bg_alpha: float = 0.5,
              padding: int = 6) -> np.ndarray:
    """
    Dibuja texto. Si se especifica bg_color, pone un fondo semitransparente.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    if bg_color is not None:
        draw_panel(
            frame,
            x - padding, y - th - padding,
            tw + padding * 2, th + baseline + padding * 2,
            bg_color, bg_alpha, radius=6
        )

    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return frame


def draw_text_centered(frame, text, cx, cy, **kwargs):
    """Versión centrada de draw_text."""
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = kwargs.get("scale", config.FONT_SCALE)
    thick = kwargs.get("thickness", config.FONT_THICKNESS)
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    draw_text(frame, text, cx - tw // 2, cy + th // 2, **kwargs)
    return frame


# ── Elementos de HUD ──────────────────────────────────────────────────────────

def draw_countdown(frame, seconds_remaining: int):
    """
    Dibuja la cuenta regresiva grande en el centro del frame.
    seconds_remaining=0 muestra "¡Sonríe!" en verde.
    """
    h, w = frame.shape[:2]
    if seconds_remaining > 0:
        text  = str(seconds_remaining)
        color = config.COLOR_ACCENT
        scale = 5.0
    else:
        text  = ":D"
        color = config.COLOR_GREEN
        scale = 4.0

    font  = cv2.FONT_HERSHEY_SIMPLEX
    thick = 8
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cx = w // 2 - tw // 2
    cy = h // 2 + th // 2

    # Sombra para legibilidad
    cv2.putText(frame, text, (cx + 3, cy + 3), font, scale, config.COLOR_BLACK, thick + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (cx, cy),          font, scale, color,             thick,     cv2.LINE_AA)
    return frame


def draw_score_hud(frame, score: int, matches: int, total: int):
    """
    Dibuja el HUD del minijuego: puntuación y progreso de pares.
    Se coloca en la esquina superior derecha.
    """
    h, w = frame.shape[:2]
    panel_w, panel_h = 220, 70
    draw_panel(frame, w - panel_w - 10, 10, panel_w, panel_h, alpha=0.65)

    draw_text(frame, f"Puntos: {score}",
              w - panel_w, 38,
              color=config.COLOR_ACCENT, scale=0.75, thickness=2)
    draw_text(frame, f"Pares: {matches}/{total}",
              w - panel_w, 68,
              color=config.COLOR_WHITE, scale=0.7, thickness=1)
    return frame


def draw_mode_indicator(frame, mode_name: str):
    """Pequeña pastilla en la esquina superior izquierda con el modo activo."""
    draw_panel(frame, 10, 10, 200, 36, alpha=0.6, radius=8)
    draw_text(frame, f"Modo: {mode_name}", 20, 34, scale=0.65, thickness=1)
    return frame


def draw_controls_help(frame, controls: dict):
    """
    Dibuja un panel de ayuda con los controles disponibles.
    controls = { "Espacio": "Foto", "N": "Siguiente sticker", ... }
    """
    h, w = frame.shape[:2]
    line_h = 26
    panel_h = len(controls) * line_h + 20
    draw_panel(frame, 10, h - panel_h - 10, 260, panel_h, alpha=0.55)

    for i, (key, action) in enumerate(controls.items()):
        y_pos = h - panel_h + 20 + i * line_h
        draw_text(frame, f"[{key}] {action}", 20, y_pos, scale=0.55, thickness=1)
    return frame


def draw_pinch_cursor(frame, point, is_pinching: bool):
    """
    Dibuja el cursor de la mano (punto medio entre pulgar e índice).
    Cambia de color si está haciendo pinch.
    """
    if point is None:
        return frame
    color  = config.COLOR_GREEN if is_pinching else config.COLOR_WHITE
    radius = 12 if is_pinching else 8
    cv2.circle(frame, point, radius,     color,               2, cv2.LINE_AA)
    cv2.circle(frame, point, radius - 5, config.COLOR_BLACK,  -1, cv2.LINE_AA)
    cv2.circle(frame, point, radius - 6, color,               -1, cv2.LINE_AA)
    return frame


def draw_connection_line(frame, p1, p2, color=config.LINE_COLOR):
    """
    Dibuja la línea de conexión del minijuego con gradiente visual simple.
    """
    if p1 is None or p2 is None:
        return frame
    # Línea gruesa con borde oscuro para legibilidad
    cv2.line(frame, p1, p2, config.COLOR_BLACK,    config.LINE_THICKNESS + 3, cv2.LINE_AA)
    cv2.line(frame, p1, p2, color,                  config.LINE_THICKNESS,     cv2.LINE_AA)
    # Puntitos en los extremos
    cv2.circle(frame, p1, 6, color, -1, cv2.LINE_AA)
    cv2.circle(frame, p2, 6, color, -1, cv2.LINE_AA)
    return frame


# ── Utilidades de carga de assets ─────────────────────────────────────────────

def load_asset(path: str) -> np.ndarray | None:
    """
    Carga una imagen PNG con canal alpha (BGRA).
    Devuelve None si el archivo no existe o no es válido.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[Overlay] Advertencia: no se pudo cargar '{path}'")
        return None

    # Si la imagen no tiene canal alpha, agregar uno completamente opaco
    if img.shape[2] == 3:
        alpha = np.ones(img.shape[:2], dtype=np.uint8) * 255
        img   = cv2.merge([img, alpha])
    return img


def placeholder_asset(w: int, h: int, color_bgr: tuple, text: str = "") -> np.ndarray:
    """
    Crea un placeholder BGRA cuando falta un asset real.
    Muy útil durante el desarrollo para no bloquear el flujo.
    """
    img     = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :, :3] = color_bgr
    img[:, :,  3] = 200   # Semi-transparente

    # Borde blanco
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (255, 255, 255, 255), 2)

    if text:
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = min(w, h) / 120.0
        (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
        cv2.putText(img, text,
                    (w // 2 - tw // 2, h // 2 + th // 2),
                    font, scale, (255, 255, 255, 255), 1, cv2.LINE_AA)
    return img

