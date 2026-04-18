# Detección y seguimiento de manos con MediaPipe
#
# Responsabilidades:
#   · Inicializar MediaPipe Hands con los parámetros de config.py
#   · Procesar cada frame y extraer landmarks normalizados y en píxeles
#   · Detectar gestos específicos: pinch (pulgar+índice), mano abierta, puño
#   · Exponer una API sencilla para que los modos no toquen MediaPipe directamente
#
# Landmarks de referencia (los más usados):
#   0  = Muñeca
#   4  = Punta del pulgar
#   8  = Punta del índice
#   12 = Punta del dedo medio
#   16 = Punta del anular
#   20 = Punta del meñique

import math
import cv2
import mediapipe as mp
import config


class HandTracker:
    """
    Wrapper sobre mediapipe.solutions.hands.Hands.

    Uso típico dentro del loop:
        tracker = HandTracker()
        ...
        results = tracker.process(frame)
        if results.hands:
            for hand in results.hands:
                tip = hand.landmarks_px[8]   # punta del índice en píxeles
                if hand.is_pinching:
                    ...
    """

    # Índices de landmarks clave (constantes para legibilidad)
    WRIST        = 0
    THUMB_TIP    = 4
    INDEX_TIP    = 8
    MIDDLE_TIP   = 12
    RING_TIP     = 16
    PINKY_TIP    = 20
    INDEX_MCP    = 5   # Nudillo base del índice (referencia de tamaño de mano)

    def __init__(self):
        self._mp_hands   = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles  = mp.solutions.drawing_styles

        self.hands = self._mp_hands.Hands(
            static_image_mode       = False,
            max_num_hands           = config.HAND_MAX_HANDS,
            model_complexity        = config.HAND_MODEL_COMPLEXITY,
            min_detection_confidence= config.HAND_DETECTION_CONF,
            min_tracking_confidence = config.HAND_TRACKING_CONF,
        )
        print("[HandTracker] MediaPipe Hands inicializado.")

    # ── Procesamiento principal ───────────────────────────────────────────────

    def process(self, frame_bgr):
        """
        Procesa un frame BGR y devuelve un objeto HandResults.

        El frame NO se modifica; usa draw_on_frame() para dibujar encima.
        """
        h, w = frame_bgr.shape[:2]

        # MediaPipe requiere RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False          # Pequeña optimización de memoria
        raw = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        hand_list = []
        if raw.multi_hand_landmarks:
            for idx, lm_list in enumerate(raw.multi_hand_landmarks):
                # Etiqueta de lateralidad ("Left" / "Right")
                label = raw.multi_handedness[idx].classification[0].label

                # Landmarks normalizados (0-1) y en píxeles
                norm_lms = [(lm.x, lm.y, lm.z) for lm in lm_list.landmark]
                px_lms   = [
                    (int(lm.x * w), int(lm.y * h))
                    for lm in lm_list.landmark
                ]

                hand_list.append(
                    HandData(
                        label        = label,
                        landmarks_norm= norm_lms,
                        landmarks_px  = px_lms,
                        raw_lm_list   = lm_list,   # para dibujar con MediaPipe
                        frame_size    = (w, h),
                    )
                )

        return HandResults(hand_list)

    # ── Dibujo de esqueleto ───────────────────────────────────────────────────

    def draw_on_frame(self, frame, results, draw_skeleton: bool = True):
        """
        Dibuja el esqueleto de manos sobre el frame (modifica en lugar).
        Útil para depuración. En producción puedes desactivarlo con draw_skeleton=False.
        """
        if not draw_skeleton or not results.hands:
            return frame

        for hand in results.hands:
            self._mp_drawing.draw_landmarks(
                frame,
                hand.raw_lm_list,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_styles.get_default_hand_landmarks_style(),
                self._mp_styles.get_default_hand_connections_style(),
            )
        return frame

    # ── Cierre limpio ─────────────────────────────────────────────────────────

    def close(self):
        self.hands.close()
        print("[HandTracker] Recursos liberados.")


# ── Clases de datos ───────────────────────────────────────────────────────────

class HandData:
    """
    Información de una mano detectada en un frame.

    Atributos:
        label           : "Left" o "Right"
        landmarks_norm  : Lista de 21 tuplas (x, y, z) normalizadas [0, 1]
        landmarks_px    : Lista de 21 tuplas (x, y) en píxeles
        frame_size      : (ancho, alto) del frame de origen
    """

    def __init__(self, label, landmarks_norm, landmarks_px, raw_lm_list, frame_size):
        self.label           = label
        self.landmarks_norm  = landmarks_norm
        self.landmarks_px    = landmarks_px
        self.raw_lm_list     = raw_lm_list
        self.frame_size      = frame_size

    # ── Gestos ────────────────────────────────────────────────────────────────

    @property
    def thumb_tip(self):
        """Punta del pulgar en píxeles."""
        return self.landmarks_px[HandTracker.THUMB_TIP]

    @property
    def index_tip(self):
        """Punta del índice en píxeles."""
        return self.landmarks_px[HandTracker.INDEX_TIP]

    @property
    def pinch_distance(self):
        """Distancia en píxeles entre pulgar e índice."""
        return _dist(self.thumb_tip, self.index_tip)

    @property
    def is_pinching(self):
        """True si el gesto pinch está activo (umbral en config.PINCH_THRESHOLD)."""
        return self.pinch_distance < config.PINCH_THRESHOLD

    @property
    def pinch_midpoint(self):
        """Punto medio entre pulgar e índice (útil como cursor del gesto)."""
        tx, ty = self.thumb_tip
        ix, iy = self.index_tip
        return ((tx + ix) // 2, (ty + iy) // 2)

    @property
    def is_open(self):
        """
        True si la mano está abierta (los 5 dedos extendidos).
        Se compara la punta de cada dedo con su segunda articulación.
        """
        tips  = [4, 8, 12, 16, 20]
        bases = [2, 6, 10, 14, 18]  # segunda articulación (pip)
        w, h  = self.frame_size

        def extended(tip_idx, base_idx):
            tip_y  = self.landmarks_norm[tip_idx][1]
            base_y = self.landmarks_norm[base_idx][1]
            # En coordenadas normalizadas, y crece hacia abajo;
            # si la punta está ARRIBA del pip, el dedo está extendido.
            return tip_y < base_y

        # El pulgar se compara en X porque dobla lateralmente
        thumb_ext = (self.landmarks_norm[4][0] >
                     self.landmarks_norm[3][0]
                     if self.label == "Right"
                     else self.landmarks_norm[4][0] < self.landmarks_norm[3][0])

        return thumb_ext and all(extended(t, b) for t, b in zip(tips[1:], bases[1:]))

    @property
    def is_fist(self):
        """
        True si la mano está cerrada (todos los dedos doblados).
        Compara la punta de cada dedo con la muñeca: si está más abajo → doblado.
        """
        tips  = [8, 12, 16, 20]
        bases = [6, 10, 14, 18]
        return all(
            self.landmarks_norm[t][1] > self.landmarks_norm[b][1]
            for t, b in zip(tips, bases)
        )

    def finger_up_count(self):
        """Cuenta cuántos dedos están extendidos (0–5)."""
        tips  = [8, 12, 16, 20]
        bases = [6, 10, 14, 18]
        count = sum(
            1 for t, b in zip(tips, bases)
            if self.landmarks_norm[t][1] < self.landmarks_norm[b][1]
        )
        # Pulgar
        thumb_up = (self.landmarks_norm[4][0] > self.landmarks_norm[3][0]
                    if self.label == "Right"
                    else self.landmarks_norm[4][0] < self.landmarks_norm[3][0])
        return count + (1 if thumb_up else 0)


class HandResults:
    """Resultado del procesamiento de un frame: lista de HandData."""

    def __init__(self, hands: list):
        self.hands = hands

    @property
    def detected(self):
        return len(self.hands) > 0

    @property
    def first(self):
        """Primera mano detectada, o None."""
        return self.hands[0] if self.hands else None

    def get_by_label(self, label: str):
        """Devuelve la HandData de la mano izquierda o derecha, o None."""
        for h in self.hands:
            if h.label == label:
                return h
        return None


# ── Utilidades ────────────────────────────────────────────────────────────────

def _dist(p1, p2):
    """Distancia euclidiana entre dos puntos (x, y)."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

