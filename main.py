# main.py

import io
import base64

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
from tensorflow import keras

# ── FastAPI app + CORS ────────────────────────────────────────────────────

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # allow all origins (you can lock this down)
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ── ChessboardLayer & loss (for model_a/model_b) ──────────────────────────

class ChessboardLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        kwargs.pop("dynamic", None)
        super().__init__(**kwargs)

    def call(self, inputs):
        # inputs: [batch, H, W, C]
        boards = tf.unstack(inputs, axis=0)
        out_boards = []
        for b in boards:
            b = tf.expand_dims(b, 0)  # [1, H, W, C]
            rows = tf.split(b, 8, axis=1)
            row_cells = []
            for r in rows:
                cells = tf.split(r, 8, axis=2)
                flat = [tf.reshape(c, [1,1,1,-1]) for c in cells]
                row_cells.append(tf.concat(flat, axis=2))  # [1,1,8,F]
            out_boards.append(tf.concat(row_cells, axis=1))  # [1,8,8,F]
        return tf.concat(out_boards, axis=0)  # [batch,8,8,F]

    def compute_output_shape(self, input_shape):
        b, H, W, C = input_shape
        cell_feat = (H*W*C)//64
        return (b, 8, 8, cell_feat)

def weighted_loss(y_true, y_pred):
    # must match how the models were trained
    weights = tf.constant([1.0,4.0,8.0,8.0,8.0,8.0,4.0], dtype=tf.float32)
    loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    idxs = tf.reshape(y_true, (-1,))
    wts  = tf.reshape(tf.gather(weights, idxs), loss.shape)
    return loss * wts

# ── Board detection & cropping from demo ─────────────────────────────────

def get_angle(ab, ac):
    cosv = (ab@ac)/(np.linalg.norm(ab)*np.linalg.norm(ac))
    return np.degrees(np.arccos(np.clip(cosv, -1, 1)))

def check_square(pts):
    if len(pts)!=4:
        return 4*90**2, 0
    a,b,c,d = np.squeeze(pts)
    angs = [
        get_angle(b-c, d-c),
        get_angle(c-d, a-d),
        get_angle(d-a, b-a),
        get_angle(a-b, c-b),
    ]
    err  = np.sum((np.array(angs)-90)**2)
    side = np.mean([
        np.linalg.norm(a-b),
        np.linalg.norm(b-c),
        np.linalg.norm(c-d),
        np.linalg.norm(d-a),
    ])
    return err, side

def child_count(i, hierarchy, is_sq):
    j = hierarchy[0,i,2]
    if j<0: return 0
    # go to first child
    while hierarchy[0,j,0]>0:
        j = hierarchy[0,j,0]
    cnt = 0
    while j>=0:
        if is_sq[j]:
            cnt += 1
        j = hierarchy[0,j,1]
    return cnt

def get_contours(img, blur):
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurimg = cv2.GaussianBlur(gray,(blur,blur),0)
    edges   = cv2.Canny(blurimg,50,150)
    dil     = cv2.dilate(edges, np.ones((3,3),np.uint8),1)
    cnts, hier = cv2.findContours(dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    approx = [cv2.approxPolyDP(c, 0.04*cv2.arcLength(c,True), True) for c in cnts]
    return approx, hier

def get_candidate_boards(cnts, hier, max_err, min_side, min_child):
    info = [check_square(c) for c in cnts]
    is_sq = [(e<max_err and s>min_side) for e,s in info]
    boards = []
    for i,((e,s),c) in enumerate(zip(info,cnts)):
        if is_sq[i] and child_count(i,hier,is_sq)>min_child:
            boards.append(c)
    return boards

def preprocess(img):
    all_b = []
    for b in (3,5,7,9,11):
        cnts, hier = get_contours(img,b)
        all_b += get_candidate_boards(cnts,hier,4e2,10,10)
    # pick the biggest (by side) within 90% of max
    scored = [(c, check_square(c)[1]) for c in all_b]
    scored.sort(key=lambda x:-x[1])
    best = (None,0)
    for c,s in scored:
        if s>=best[1]*0.9:
            best=(c,s)
    return best[0]

def crop_image(img, pts):
    pts = np.squeeze(pts)
    x0,x1 = pts[:,0].min(), pts[:,0].max()
    y0,y1 = pts[:,1].min(), pts[:,1].max()
    return img[y0:y1, x0:x1]

# ── Ensemble loading ──────────────────────────────────────────────────────

LOAD = [
    ("model/model_a.h5", 0.5),
    ("model/model_b.h5", 0.5),
]

_models = [
    keras.models.load_model(
        path,
        custom_objects={"ChessboardLayer":ChessboardLayer,
                        "weighted_loss":weighted_loss},
        compile=False
    )
    for path,_ in LOAD
]

# Map index → piece letter (demo mapping)
_index_to_letter = {
    0:'-',   # empty
    1:'N',2:'B',3:'R',4:'Q',5:'K',6:'P'
}

def vector_to_class(grid):
    vec = np.vectorize(lambda i: _index_to_letter[int(i)])
    return vec(grid)

def letters_to_fen(letter_grid):
    rows=[]
    for r in range(8):
        empties=0; row=""
        for c in letter_grid[r]:
            if c=='-':
                empties+=1
            else:
                if empties:
                    row+=str(empties)
                    empties=0
                row+=c
        if empties:
            row+=str(empties)
        rows.append(row)
    return "/".join(rows)+" w - - 0 1"

# ── The POST /scan endpoint ────────────────────────────────────────────────

@app.post("/scan")
async def scan_chessboard(file: UploadFile = File(...)):
    # 1) Read & decode
    data = await file.read()
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # 2) Detect & crop → warp
    pts     = preprocess(img)
    cropped = crop_image(img, pts)
    cropped = cv2.resize(cropped, (512,512), interpolation=cv2.INTER_AREA)

    # 3) Grayscale normalize & shape= (1,512,512,1)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    inp  = (gray.astype(np.float32)/255.0)[...,None][None,...]

    # 4) Ensemble predict
    y_acc = np.zeros((1,64,7), dtype=np.float32)
    for m,( _, w ) in zip(_models, LOAD):
        y_acc += m.predict(inp) * w

    grid = np.argmax(y_acc, axis=-1).reshape(8,8)  # ints 0–6

    # 5) To letters & FEN
    letters = vector_to_class(grid)
    fen     = letters_to_fen(letters)

    # 6) Overlay letters on FFT→BGR
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i in range(8):
        for j in range(8):
            cv2.putText(overlay, letters[i,j],
                        (j*64+5, i*64+50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 2)

    # 7) Encode to PNG/base64
    _, buf = cv2.imencode(".png", overlay)
    b64     = base64.b64encode(buf).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64}"

    # 8) Return
    return {"fen": fen, "image": data_uri}
