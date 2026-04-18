# Punto de entrada del proyecto OpenSource Booth
#
# Responsabilidades:
#   · Inicializar la cámara, el hand tracker y los dos modos
#   · Ejecutar el loop principal de captura y procesamiento
#   · Enrutar teclas hacia el modo activo
#   · Cerrar todos los recursos limpiamente al salir
#
# Uso:
#   uv run python main.py
#
# Controles globales:
#   Q       → Salir
#   1       → Cambiar a modo Photo Booth
#   2       → Cambiar a modo Minijuego

import sys
import cv2

import config
from core.camera       import Camera
from core.hand_tracker import HandTracker
from core.overlay      import draw_mode_indicator, draw_text
from modes.photo_booth import PhotoBooth
from modes.match_game  import MatchGame


def main():
    print("=" * 50)
    print("  OpenSource Booth — iniciando...")
    print("  [1] Photo Booth   [2] Minijuego   [Q] Salir")
    print("=" * 50)

    # ── Inicialización ────────────────────────────────────────────────────────
    try:
        camera = Camera()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    hand_tracker = HandTracker()
    photo_booth  = PhotoBooth()
    match_game   = MatchGame()

    # Modo inicial
    current_mode = "booth"   # "booth" | "game"

    # ── Loop principal ────────────────────────────────────────────────────────
    print("\n[Main] Loop iniciado. Presiona Q para salir.")

    while True:
        # 1. Leer frame de la cámara
        ret, frame = camera.read()
        if not ret:
            print("[Main] No se pudo leer el frame. Verificar cámara.")
            break

        # 2. Procesar manos (usado por ambos modos)
        hand_results = hand_tracker.process(frame)

        # 3. Actualizar el modo activo
        if current_mode == "booth":
            display = photo_booth.update(frame)
        else:
            display = match_game.update(frame, hand_results)

        # 4. Indicador de modo en pantalla
        mode_label = "Photo Booth" if current_mode == "booth" else "Minijuego OSS"
        draw_mode_indicator(display, mode_label)

        # 5. Mostrar frame y capturar tecla
        key = camera.show(display)

        # 6. Teclas globales (siempre activas)
        if key == config.KEY_QUIT:
            print("[Main] Saliendo por tecla Q.")
            break

        elif key == config.KEY_MODE_BOOTH:
            if current_mode != "booth":
                current_mode = "booth"
                print("[Main] Modo: Photo Booth")

        elif key == config.KEY_MODE_GAME:
            if current_mode != "game":
                current_mode = "game"
                print("[Main] Modo: Minijuego")

        # 7. Delegar teclas al modo activo
        elif key != 255 and key != -1:   # 255/−1 = sin tecla presionada
            if current_mode == "booth":
                action = photo_booth.handle_key(key)
            else:
                action = match_game.handle_key(key)

            # El modo puede pedir un cambio de modo o salir
            if action == "quit":
                break
            elif action == "mode_game":
                current_mode = "game"
                print("[Main] Modo: Minijuego (desde Photo Booth)")
            elif action == "mode_booth":
                current_mode = "booth"
                print("[Main] Modo: Photo Booth (desde Minijuego)")

    # ── Cierre limpio ─────────────────────────────────────────────────────────
    print("[Main] Cerrando recursos...")
    photo_booth.close()
    match_game.close()
    hand_tracker.close()
    camera.release()
    print("[Main] Hasta luego.")


if __name__ == "__main__":
    main()
