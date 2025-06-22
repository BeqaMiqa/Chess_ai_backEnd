# main.py

import io
import base64

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras

# ── FastAPI + CORS setup ─────────────────────────────────────────────────

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",              # your dev front-end
        "https://your-production-domain.com", # add your real domain here
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ── Board Detection & Cropping Helpers ────────────────────────────────────

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
    # go to first child
    while hierarchy[0, j, 0] > 0:
        j = hierarchy[0, j, 0]
    count = 0
    # count all siblings that are squares
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
    cnts, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    approx = [cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True) for c in cnts]
    return approx, hierarchy

def get_candidate_boards(contours, hierarchy, max_error, min_side, min_child):
    info      = [check_square(c) for c in contours]
    is_square = [(e < max_error and s > min_side) for e, s in info]
    boards    = []
    for i, ((e, s), cnt) in enumerate(zip(info, contours)):
        if is_square[i] and child_count(i, hierarchy, is_square) > min_child:
            boards.append(cnt)
    return boards

def preprocess(image):
    all_boards = []
    for blur in (3, 5, 7, 9, 11):
        cnts, hier = get_contours(image, blur)
        cands      = get_candidate_boards(cnts, hier, 4e2, 10, 10)
        all_boards += cands
    # pick the board whose side‐length is within 90% of max
    scored = [(b, check_square(b)[1]) for b in all_boards]
    scored.sort(key=lambda x: x[1], reverse=True)
    best = (None, 0)
    for board, side in scored:
        if side >= best[1] * 0.9:
            best = (board, side)
    return best[0]

def crop_image(image, pts):
    pts = np.squeeze(pts)
    x0, x1 = pts[:,0].min(), pts[:,0].max()
    y0, y1 = pts[:,1].min(), pts[:,1].max()
    return image[y0:y1, x0:x1]

# ── FEN Conversion Helpers ─────────────────────────────────────────────────

_index_to_letter = {
    0:'p',1:'n',2:'b',3:'r',4:'q',5:'k',
    6:'P',7:'N',8:'B',9:'R',10:'Q',11:'K',
    12:'-'  # empty
}

def vector_to_class(grid):
    vec = np.vectorize(lambda idx: _index_to_letter[int(idx)])
    return vec(grid)

def letters_to_fen(letter_grid):
    rows = []
    for r in range(8):
        empties = 0
        row     = ""
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

# ── Load your two small models once at startup ─────────────────────────────

# adjust these paths if your models live elsewhere
LOAD = [
    ("model/model_a.h5", 0.5),
    ("model/model_b.h5", 0.5),
]

_models = [keras.models.load_model(path, compile=False) for path,_ in LOAD]
label_cnt = _models[0].output_shape[-1]  # e.g. 13

# ── Main endpoint ─────────────────────────────────────────────────────────

@app.post("/scan")
async def scan_chessboard(file: UploadFile = File(...)):
    # 1) Read uploaded bytes into OpenCV BGR array
    data = await file.read()
    arr  = np.frombuffer(data, dtype=np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # 2) Detect & crop the board, warp to 512×512
    pts     = preprocess(img)
    cropped = crop_image(img, pts)
    cropped = cv2.resize(cropped, (512, 512), interpolation=cv2.INTER_AREA)

    # 3) Prepare 3-channel input for ensemble
    inp_color = cropped.astype(np.float32) / 255.0    # [512,512,3]
    inp_color = np.expand_dims(inp_color, 0)          # [1,512,512,3]

    # 4) Ensemble predict
    acc = np.zeros((1, 64, label_cnt), dtype=np.float32)
    for model, weight in zip(_models, [w for _,w in LOAD]):
        acc += model.predict(inp_color) * weight      # each → [1,64,label_cnt]

    grid    = np.argmax(acc, axis=-1).reshape(8, 8)   # [8,8] ints
    letters = vector_to_class(grid)
    fen     = letters_to_fen(letters)

    # 5) Build gray overlay for visualization
    gray    = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i in range(8):
        for j in range(8):
            cv2.putText(
                overlay, letters[i,j],
                (j*64 + 5, i*64 + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0,255,255), 2
            )

    # 6) Encode PNG → base64
    _, buf   = cv2.imencode(".png", overlay)
    data_uri = "data:image/png;base64," + base64.b64encode(buf).decode()

    # 7) Return JSON
    return {"fen": fen, "image": data_uri}
