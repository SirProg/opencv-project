# core/camera.py — Captura de cámara y loop principal de frames
#
# Responsabilidades:
#   · Abrir y configurar el dispositivo de captura (cv2.VideoCapture)
#   · Leer frames y aplicar el flip horizontal si está activado
#   · Proveer el frame actual a los módulos que lo necesiten
#   · Mostrar el frame final en pantalla y manejar el cierre limpio
#
# Uso:
#   cam = Camera()
#   ret, frame = cam.read()
#   cam.show(frame)
#   cam.release()

import cv2
import config


class Camera:
    """Wrapper sobre cv2.VideoCapture con configuración automática."""

    def __init__(self):
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)

        # Configurar resolución y FPS en el dispositivo
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          config.CAMERA_FPS)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la cámara con índice {config.CAMERA_INDEX}. "
                "Verifica que esté conectada o cambia CAMERA_INDEX en config.py"
            )

        # Leer resolución real que negoció el driver (puede diferir de la pedida)
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] Resolución activa: {self.width}×{self.height}")

    # ── Lectura de frame ──────────────────────────────────────────────────────

    def read(self):
        """
        Lee el siguiente frame de la cámara.

        Returns:
            (bool, np.ndarray | None) — éxito y frame BGR.
            Si ret=False, frame=None y la cámara puede haberse desconectado.
        """
        ret, frame = self.cap.read()
        if not ret:
            return False, None

        if config.FLIP_HORIZONTAL:
            frame = cv2.flip(frame, 1)

        return True, frame

    # ── Mostrar frame ─────────────────────────────────────────────────────────

    def show(self, frame, window_name: str = "OpenSource Booth"):
        """
        Muestra el frame en una ventana de OpenCV.
        Devuelve la tecla presionada (o -1 si no hubo ninguna).
        Espera 1 ms para no bloquear el loop.
        """
        cv2.imshow(window_name, frame)
        return cv2.waitKey(1) & 0xFF

    # ── Cierre limpio ─────────────────────────────────────────────────────────

    def release(self):
        """Libera el dispositivo de captura y destruye todas las ventanas."""
        self.cap.release()
        cv2.destroyAllWindows()
        print("[Camera] Cámara liberada correctamente.")

    # ── Propiedades útiles ────────────────────────────────────────────────────

    @property
    def size(self):
        """Tupla (ancho, alto) de la resolución activa."""
        return (self.width, self.height)

    @property
    def center(self):
        """Centro del frame (x, y)."""
        return (self.width // 2, self.height // 2)
