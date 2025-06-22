# main.py

import base64
import io

import numpy as np
import cv2
from PIL import Image

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from tensorflow import keras

# ── FastAPI + CORS setup ─────────────────────────────────────────────────

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten this in production!
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ── Board Detection & Cropping Helpers (from your demo2.ipynb) ────────────

def get_angle(ab, ac):
    cosv = (ab @ ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    cosv = np.clip(cosv, -1, 1)
    return np.degrees(np.arccos(cosv))

def check_square(pts):
    if len(pts) != 4:
        return 4 * 90**2, 0
    a, b, c, d = np.squeeze(pts)
    angles = [
        get_angle(b - c, d - c),
        get_angle(c - d, a - d),
        get_angle(d - a, b - a),
        get_angle(a - b, c - b),
    ]
    error = np.sum((np.array(angles) - 90)**2)
    side  = np.mean([
        np.linalg.norm(a - b),
        np.linalg.norm(b - c),
        np.linalg.norm(c - d),
        np.linalg.norm(d - a),
    ])
    return error, side

def child_count(i, hierarchy, is_square):
    j = hierarchy[0, i, 2]
    if j < 0:
        return 0
    while hierarchy[0, j, 0] > 0:
        j = hierarchy[0, j, 0]
    count = 0
    while j >= 0:
        if is_square[j]:
            count += 1
        j = hierarchy[0, j, 1]
    return count

def get_contours(image, blur_radius):
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_radius, blur_radius), 0)
    edges   = cv2.Canny(blurred, 50, 150)
    dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    cnts, hierarchy = cv2.findContours(
        dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    approx = [
        cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        for c in cnts
    ]
    return approx, hierarchy

def get_candidate_boards(contours, hierarchy, max_error, min_side, min_child):
    sq_info   = [check_square(c) for c in contours]
    is_square = [(e < max_error and s > min_side) for e, s in sq_info]
    boards    = []
    for i, ((e, s), cnt) in enumerate(zip(sq_info, contours)):
        if is_square[i] and child_count(i, hierarchy, is_square) > min_child:
            boards.append(cnt)
    return boards

def preprocess(image):
    all_boards = []
    for blur in (3,5,7,9,11):
        contours, hierarchy = get_contours(image, blur)
        candidates = get_candidate_boards(contours, hierarchy, 4e2, 10, 10)
        all_boards.extend(candidates)
    scored = [(b, check_square(b)[1]) for b in all_boards]
    scored.sort(key=lambda x: x[1], reverse=True)
    best = (None, 0)
    for board, side in scored:
        if side >= best[1] * 0.9:
            best = (board, side)
    return best[0]

def crop_image(image, points):
    pts = np.squeeze(points)
    x0, x1 = pts[:,0].min(), pts[:,0].max()
    y0, y1 = pts[:,1].min(), pts[:,1].max()
    return image[y0:y1, x0:x1]


# ── Model Ensemble Helpers ────────────────────────────────────────────────

# Map index → piece letter
_index_to_letter = {
    0:'p',1:'n',2:'b',3:'r',4:'q',5:'k',
    6:'P',7:'N',8:'B',9:'R',10:'Q',11:'K',
    12:'-'  # empty square
}

def vector_to_class(grid):
    vec = np.vectorize(lambda idx: _index_to_letter[int(idx)])
    return vec(grid)

def letters_to_fen(letter_grid):
    rows = []
    for r in range(8):
        empties, row = 0, ""
        for c in letter_grid[r]:
            if c == '-':
                empties += 1
            else:
                if empties:
                    row += str(empties)
                    empties = 0
                row += c
        if empties:
            row += str(empties)
        rows.append(row)
    return "/".join(rows) + " w - - 0 1"

# Load the two small models with equal weight
LOAD = [
    ("model/model_a.h5", 0.5),
    ("model/model_b.h5", 0.5),
]
_models = [keras.models.load_model(path, compile=False) for path,_ in LOAD]

def ensemble_predict(inp: np.ndarray) -> np.ndarray:
    """Run both models and return weighted average predictions."""
    accum = np.zeros((1, 64, 13), dtype=np.float32)
    for m, w in zip(_models, [w for _,w in LOAD]):
        p = m.predict(inp)  # (1,64,13)
        accum += p * w
    return accum


# ── FastAPI Route ────────────────────────────────────────────────────────

@app.post("/scan")
async def scan_chessboard(file: UploadFile = File(...)):
    # 1) Read & decode image
    data = await file.read()
    arr  = np.frombuffer(data, dtype=np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # 2) Detect & crop the board
    pts     = preprocess(img)
    cropped = crop_image(img, pts)
    cropped = cv2.resize(cropped, (512,512), interpolation=cv2.INTER_AREA)

    # 3) Grayscale & normalize
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    inp  = (gray.astype(np.float32)/255.0)[...,None][None,...]

    # 4) Ensemble predict
    preds = ensemble_predict(inp)         # (1,64,13)
    grid  = np.argmax(preds, axis=-1).reshape(8,8)

    # 5) To letters & FEN
    letters = vector_to_class(grid)
    fen     = letters_to_fen(letters)

    # 6) Overlay letters on the gray image
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i in range(8):
        for j in range(8):
            cv2.putText(
                overlay, letters[i,j],
                (j*64+5, i*64+50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0,255,255), 2
            )

    # 7) Encode PNG → base64
    _, buf   = cv2.imencode(".png", overlay)
    data_uri = "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")

    # 8) Return JSON payload
    return {
        "fen": fen,
        "image": data_uri
    }
