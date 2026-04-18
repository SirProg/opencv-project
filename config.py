# config.py - Global Configuration

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
FLIP_HORIZONTAL = True

#------------------------

# MdeiaPipe - Hands
HAND_MAX_HANDS = 2
HAND_DETECTION_CONF = 0.7
HAND_TRACKING_CONF = 0.6
HAND_MODEL_COMPLEXITY = 0

#------------------------

# MediaPipe - Faces
FACE_MAX_FACES = 4
FACE_DETECTION_CONF = 0.5
FACE_TRACKING_CONF = 0.5
FACE_REFINE_LANDMARKS = True

#------------------------

# Photo Booth
PHOTOS_OUTPUT_DIR = "pictures"
PHOTO_COUNTDOWN = 3
STICKER_SCALE_FACTOR = 1.4

#------------------------

# Minigame - match 
PINCH_THRESHOLD = 40
LINE_COLOR = (255, 20, 0)
LINE_THICKNESS = 3
MATCH_RADIUS = 60 
SCORE_PER_MATCH = 100
TOTAL_PAIRS = 5 

#------------------------

# UI 
FONT = 0 
FONT_SCALE = 0.8
FONT_THICKNESS = 2 
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GREEN = (0, 220, 80)
COLOR_RED = (0, 60, 220)
COLOR_ACCENT = (255, 180, 0)
COLOR_UI_BG = (20, 20, 20)
UI_ALPHA = 0.6

#------------------------

# Control Key
KEY_QUIT = ord('q')
KEY_MODE_BOOTH = ord('1')
KEY_MODE_GAME = ord('2')
KEY_CAPTURE_PHOTO = ord(' ')
KEY_NEXT_STICKER = ord('n')


