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
        "http://localhost:8080",            # your React dev server
        "https://your-production-domain.com" # your real frontend URL
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ── Board detection & cropping helpers (from demo2.ipynb) ─────────────────

def get_angle(ab, ac):
    cosv = (ab @ ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    cosv = np.clip(cosv, -1, 1)
    return np.degrees(np.arccos(cosv))

def check_square(pts):
    if len(pts) != 4:
        return 4*90**2, 0
    a, b, c, d = np.squeeze(pts)
    angles = [
        get_angle(b-c, d-c),
        get_angle(c-d, a-d),
        get_angle(d-a, b-a),
        get_angle(a-b, c-b),
    ]
    error = np.sum((np.array(angles)-90)**2)
    side  = np.mean([
        np.linalg.norm(a-b),
        np.linalg.norm(b-c),
        np.linalg.norm(c-d),
        np.linalg.norm(d-a),
    ])
    return error, side

def child_count(i, hierarchy, is_sq):
    j = hierarchy[0,i,2]
    # find first child index
    while j>=0 and hierarchy[0,j,0]>0:
        j = hierarchy[0,j,0]
    cnt = 0
    # count all square siblings
    while j>=0:
        if is_sq[j]:
            cnt += 1
        j = hierarchy[0,j,1]
    return cnt

def get_contours(img, blur):
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur,blur), 0)
    edges   = cv2.Canny(blurred, 50,150)
    dilated = cv2.dilate(edges, np.ones((3,3),np.uint8), iterations=1)
    cnts, hier = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    approx = [cv2.approxPolyDP(c, 0.04*cv2.arcLength(c,True), True) for c in cnts]
    return approx, hier

def get_candidate_boards(cnts, hier, max_err, min_side, min_child):
    info     = [check_square(c) for c in cnts]
    is_sq    = [(e<max_err and s>min_side) for e,s in info]
    boards   = []
    for i, ((e,s),c) in enumerate(zip(info, cnts)):
        if is_sq[i] and child_count(i, hier, is_sq)>min_child:
            boards.append((c, s))
    # keep only the largest one
    boards.sort(key=lambda x: -x[1])
    return [b for b,_ in boards[:1]]

def preprocess(img):
    all_boards = []
    for blur in (3,5,7,9,11):
        cnts, hier = get_contours(img, blur)
        cands      = get_candidate_boards(cnts, hier, 4e2, 10,10)
        all_boards += cands
    # pick best side length within 90% of max
    scored = [(b, check_square(b)[1]) for b in all_boards]
    scored.sort(key=lambda x:-x[1])
    best = (None,0)
    for b,s in scored:
        if s >= best[1]*0.9:
            best = (b,s)
    return best[0]

def crop_image(img, pts):
    pts = np.squeeze(pts)
    x0,x1 = pts[:,0].min(), pts[:,0].max()
    y0,y1 = pts[:,1].min(), pts[:,1].max()
    return img[y0:y1, x0:x1]

# ── FEN / letter mapping helpers ───────────────────────────────────────────

INDEX_TO_LETTER = {
    0:'p',1:'n',2:'b',3:'r',4:'q',5:'k',
    6:'P',7:'N',8:'B',9:'R',10:'Q',11:'K',
    12:'-'  # empty
}
def vector_to_class(grid):
    vec = np.vectorize(lambda idx: INDEX_TO_LETTER[int(idx)])
    return vec(grid)

def letters_to_fen(grid):
    rows = []
    for r in grid:
        empties = 0
        fen_r   = ""
        for c in r:
            if c=='-':
                empties +=1
            else:
                if empties:
                    fen_r += str(empties)
                    empties = 0
                fen_r += c
        if empties:
            fen_r += str(empties)
        rows.append(fen_r)
    return "/".join(rows) + " w - - 0 1"

# ── Load your two small models into memory ─────────────────────────────────

LOAD = [
    ("model/model_a.h5", 0.5),
    ("model/model_b.h5", 0.5),
]
_models = [keras.models.load_model(path, compile=False) for path,_ in LOAD]
_weights = [w for _,w in LOAD]

def ensemble_predict(x):
    """ x shaped (1,512,512,1) → returns (1,64,13) """
    acc = np.zeros((1,64,13),dtype=np.float32)
    for m,w in zip(_models,_weights):
        acc += m.predict(x) * w
    return acc

# ── The /scan endpoint ─────────────────────────────────────────────────────

@app.post("/scan")
async def scan_chessboard(file: UploadFile = File(...)):
    # 1) read bytes → BGR image
    data = await file.read()
    arr  = np.frombuffer(data, dtype=np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # 2) detect & crop
    pts     = preprocess(img)
    cropped = crop_image(img, pts)
    cropped = cv2.resize(cropped, (512,512), interpolation=cv2.INTER_AREA)

    # 3) gray + normalize + shape for CNN
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    inp  = (gray.astype(np.float32)/255.0)[...,None][None,...]  # (1,512,512,1)

    # 4) ensemble predict, argmax → 8×8 grid of ints
    preds   = ensemble_predict(inp)               # (1,64,13)
    indices = np.argmax(preds,axis=-1).reshape(8,8)  # 8×8

    # 5) to letters + FEN
    letters = vector_to_class(indices)
    fen     = letters_to_fen(letters)

    # 6) overlay letters on gray board
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i in range(8):
        for j in range(8):
            cv2.putText(overlay, letters[i,j],
                        (j*64+5, i*64+50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0,255,255), 2)

    # 7) PNG→base64
    _, buf   = cv2.imencode(".png", overlay)
    b64      = base64.b64encode(buf).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64}"

    return {"fen": fen, "image": data_uri}
