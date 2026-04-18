# Modo Photo Booth
#
# Responsabilidades:
#   · Detectar rostros con MediaPipe FaceMesh y anclar stickers encima de la cabeza
#   · Permitir al usuario seleccionar stickers con tecla 'N'
#   · Gestionar la cuenta regresiva y guardar la foto sin UI visible
#   · Soportar múltiples personas en el mismo frame
#
# Flujo:
#   1. El usuario elige un sticker con [N]
#   2. El sticker aparece encima de cada cara detectada en tiempo real
#   3. Presiona [Espacio] → cuenta regresiva → foto guardada sin UI
#
# Sobre los stickers:
#   · Coloca tus PNG con alpha en assets/stickers/
#   · El nombre del archivo se usa como etiqueta en el selector
#   · Si no hay assets, se generan placeholders de colores
#
# Landmarks de FaceMesh usados:
#   · Top de la cabeza: se estima proyectando desde el centro entre cejas (punto 8)
#     hacia arriba con el alto del bounding box como referencia.

import os
import time
import cv2
import mediapipe as mp
import numpy as np

import config
from core.overlay import (
    overlay_image, draw_panel, draw_text, draw_text_centered,
    draw_countdown, draw_controls_help, load_asset, placeholder_asset
)


# Landmarks de FaceMesh relevantes para el posicionamiento del sombrero
_LM_FOREHEAD_TOP   = 10    # Frente superior (referencia vertical)
_LM_CHIN           = 152   # Mentón (para calcular alto de cara)
_LM_LEFT_EAR       = 234   # Para calcular ancho de cara
_LM_RIGHT_EAR      = 454


class PhotoBooth:
    """
    Gestiona el modo Photo Booth completo.
    Se instancia una sola vez y se llama a update() en cada frame del loop.
    """

    CONTROLS = {
        "N":      "Siguiente sticker",
        "Espacio":"Tomar foto",
        "1/2":    "Cambiar modo",
        "Q":      "Salir",
    }

    def __init__(self):
        # ── MediaPipe FaceMesh ────────────────────────────────────────────────
        self._mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh     = self._mp_face_mesh.FaceMesh(
            static_image_mode        = False,
            max_num_faces            = config.FACE_MAX_FACES,
            refine_landmarks         = config.FACE_REFINE_LANDMARKS,
            min_detection_confidence = config.FACE_DETECTION_CONF,
            min_tracking_confidence  = config.FACE_TRACKING_CONF,
        )
        print("[PhotoBooth] MediaPipe FaceMesh inicializado.")

        # ── Stickers ──────────────────────────────────────────────────────────
        self.stickers       = []   # Lista de arrays BGRA
        self.sticker_names  = []   # Nombres para el selector
        self.current_idx    = 0    # Índice del sticker seleccionado
        self._load_stickers()

        # ── Estado de captura ─────────────────────────────────────────────────
        self._counting_down  = False
        self._countdown_end  = 0.0
        self._last_clean_frame = None   # Frame sin UI para guardar

        # Crear carpeta de salida si no existe
        os.makedirs(config.PHOTOS_OUTPUT_DIR, exist_ok=True)

    # ── API pública ───────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> np.ndarray:
        """
        Procesa un frame y devuelve dos versiones:
            · frame_display : con stickers + UI superpuesta (para mostrar en pantalla)
            · frame_clean   : con stickers pero SIN UI (para guardar como foto)

        Returns:
            np.ndarray — frame_display listo para mostrar.
        """
        # 1. Detectar caras
        faces = self._detect_faces(frame)

        # 2. Componer el frame limpio (stickers sobre la cámara, sin UI)
        clean = frame.copy()
        sticker = self.stickers[self.current_idx] if self.stickers else None
        for face in faces:
            clean = self._draw_sticker_on_face(clean, face, sticker)
        self._last_clean_frame = clean

        # 3. Construir el frame de display añadiendo la UI encima del limpio
        display = clean.copy()
        display = self._draw_ui(display, len(faces))

        # 4. Manejar cuenta regresiva
        if self._counting_down:
            remaining = self._countdown_end - time.time()
            if remaining > 0:
                display = draw_countdown(display, int(remaining) + 1)
            else:
                display = draw_countdown(display, 0)
                self._save_photo()
                self._counting_down = False

        return display

    def handle_key(self, key: int) -> str | None:
        """
        Procesa teclas específicas de este modo.
        Devuelve None o un string de acción para main.py ('quit', 'mode_game').
        """
        if key == config.KEY_NEXT_STICKER:
            self.current_idx = (self.current_idx + 1) % max(len(self.stickers), 1)
            print(f"[PhotoBooth] Sticker: {self.sticker_names[self.current_idx]}")

        elif key == config.KEY_CAPTURE_PHOTO and not self._counting_down:
            self._start_countdown()

        elif key == config.KEY_MODE_GAME:
            return "mode_game"

        elif key == config.KEY_QUIT:
            return "quit"

        return None

    def close(self):
        self.face_mesh.close()
        print("[PhotoBooth] FaceMesh liberado.")

    # ── Detección de caras ────────────────────────────────────────────────────

    def _detect_faces(self, frame: np.ndarray) -> list:
        """
        Devuelve una lista de dicts con los datos de posicionamiento de cada cara:
            {
                "cx": int,         # Centro horizontal de la cara en píxeles
                "forehead_y": int, # Y del top de la frente (para anclar el sombrero)
                "face_w": int,     # Ancho estimado de la cara
                "face_h": int,     # Alto estimado de la cara
            }
        """
        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        faces  = []

        if not result.multi_face_landmarks:
            return faces

        for face_lms in result.multi_face_landmarks:
            lms = face_lms.landmark

            # Puntos en píxeles
            def px(idx):
                return int(lms[idx].x * w), int(lms[idx].y * h)

            top_pt   = px(_LM_FOREHEAD_TOP)
            chin_pt  = px(_LM_CHIN)
            left_pt  = px(_LM_LEFT_EAR)
            right_pt = px(_LM_RIGHT_EAR)

            face_h = chin_pt[1] - top_pt[1]
            face_w = abs(right_pt[0] - left_pt[0])
            cx     = (left_pt[0] + right_pt[0]) // 2

            # El sombrero se ancla ligeramente por ENCIMA del top de la frente
            forehead_y = top_pt[1] - int(face_h * 0.15)

            faces.append({
                "cx":         cx,
                "forehead_y": forehead_y,
                "face_w":     face_w,
                "face_h":     face_h,
            })

        return faces

    # ── Dibujar sticker sobre una cara ────────────────────────────────────────

    def _draw_sticker_on_face(self, frame, face: dict, sticker) -> np.ndarray:
        """
        Coloca el sticker seleccionado encima de la cara.
        El ancho del sticker escala con el ancho de la cara.
        """
        if sticker is None:
            return frame

        # Escalar sticker proporcionalmente al ancho de la cara
        target_w = int(face["face_w"] * config.STICKER_SCALE_FACTOR)
        sh, sw   = sticker.shape[:2]
        target_h = int(target_w * sh / sw)

        # Posición: centrado en cx, su borde inferior toca la frente
        x = face["cx"] - target_w // 2
        y = face["forehead_y"] - target_h

        return overlay_image(frame, sticker, x, y, target_w, target_h)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _draw_ui(self, frame: np.ndarray, num_faces: int) -> np.ndarray:
        """Dibuja todos los elementos de UI del Photo Booth."""
        h, w = frame.shape[:2]

        # Panel de stickers (selector en la parte superior)
        self._draw_sticker_selector(frame)

        # Indicador de caras detectadas
        face_text = f"{num_faces} persona{'s' if num_faces != 1 else ''} detectada{'s' if num_faces != 1 else ''}"
        draw_text(frame, face_text, 10, h - 80,
                  color=config.COLOR_GREEN if num_faces > 0 else config.COLOR_WHITE,
                  scale=0.65, thickness=1,
                  bg_color=config.COLOR_UI_BG, bg_alpha=0.5)

        # Controles
        draw_controls_help(frame, self.CONTROLS)

        return frame

    def _draw_sticker_selector(self, frame: np.ndarray):
        """
        Barra superior con miniaturas de stickers disponibles.
        El sticker activo se destaca con un borde de color.
        """
        if not self.stickers:
            draw_text(frame, "Coloca PNGs en assets/stickers/", 10, 40,
                      scale=0.6, color=config.COLOR_ACCENT,
                      bg_color=config.COLOR_UI_BG)
            return

        thumb_size = 64
        padding    = 10
        x_start    = 10
        y_start    = 10

        draw_panel(frame, x_start - 5, y_start - 5,
                   len(self.stickers) * (thumb_size + padding) + 10,
                   thumb_size + 20, alpha=0.6)

        for i, sticker in enumerate(self.stickers):
            x = x_start + i * (thumb_size + padding)
            y = y_start + 4

            # Miniatura del sticker
            thumb_bgra = cv2.resize(sticker, (thumb_size, thumb_size))
            frame = overlay_image(frame, thumb_bgra, x, y, thumb_size, thumb_size)

            # Borde de selección
            color  = config.COLOR_ACCENT if i == self.current_idx else config.COLOR_WHITE
            thick  = 3 if i == self.current_idx else 1
            cv2.rectangle(frame, (x - 2, y - 2),
                          (x + thumb_size + 2, y + thumb_size + 2),
                          color, thick)

        return frame

    # ── Countdown y captura ───────────────────────────────────────────────────

    def _start_countdown(self):
        self._counting_down = True
        self._countdown_end = time.time() + config.PHOTO_COUNTDOWN
        print(f"[PhotoBooth] Cuenta regresiva iniciada ({config.PHOTO_COUNTDOWN}s)...")

    def _save_photo(self):
        """Guarda el frame limpio (sin UI) con timestamp en el nombre."""
        if self._last_clean_frame is None:
            return
        ts       = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(config.PHOTOS_OUTPUT_DIR, f"foto_{ts}.jpg")
        cv2.imwrite(filename, self._last_clean_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[PhotoBooth] Foto guardada: {filename}")

    # ── Carga de assets ───────────────────────────────────────────────────────

    def _load_stickers(self):
        """
        Carga todos los PNG de assets/stickers/.
        Si no hay archivos, crea placeholders de colores para poder desarrollar.
        """
        sticker_dir = os.path.join("assets", "stickers")
        os.makedirs(sticker_dir, exist_ok=True)

        png_files = sorted(
            f for f in os.listdir(sticker_dir) if f.lower().endswith(".png")
        )

        if png_files:
            for fn in png_files:
                path = os.path.join(sticker_dir, fn)
                img  = load_asset(path)
                if img is not None:
                    self.stickers.append(img)
                    self.sticker_names.append(os.path.splitext(fn)[0])
            print(f"[PhotoBooth] {len(self.stickers)} sticker(s) cargados: {self.sticker_names}")
        else:
            # Placeholders de desarrollo
            placeholders = [
                ((60, 60, 180),  "RedHat"),
                ((200, 100, 0),  "Tux"),
                ((0, 120, 0),    "GNU"),
                ((0, 0, 180),    "Terminal"),
            ]
            for color, name in placeholders:
                self.stickers.append(placeholder_asset(128, 128, color, name))
                self.sticker_names.append(name)
            print(f"[PhotoBooth] Usando {len(self.stickers)} placeholder(s). "
                  "Agrega PNGs reales a assets/stickers/")

