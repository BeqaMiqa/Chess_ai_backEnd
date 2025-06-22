import base64
import numpy as np
import cv2

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from tensorflow import keras
import tensorflow as tf

# ── FastAPI + CORS setup ─────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in prod!
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ── Board detection helpers (unchanged) ──────────────────────────────────
def get_angle(ab, ac):
    cosv = (ab @ ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    cosv = np.clip(cosv, -1, 1)
    return np.degrees(np.arccos(cosv))

def check_square(pts):
    if len(pts) != 4:
        return 4 * 90**2, 0
    a,b,c,d = np.squeeze(pts)
    angles = [
        get_angle(b-c, d-c),
        get_angle(c-d, a-d),
        get_angle(d-a, b-a),
        get_angle(a-b, c-b),
    ]
    error = np.sum((np.array(angles)-90)**2)
    side  = np.mean([np.linalg.norm(a-b), np.linalg.norm(b-c),
                     np.linalg.norm(c-d), np.linalg.norm(d-a)])
    return error, side

def child_count(i, hierarchy, is_square):
    j = hierarchy[0,i,2]
    if j < 0: return 0
    while hierarchy[0,j,0] > 0:
        j = hierarchy[0,j,0]
    cnt = 0
    while j >= 0:
        if is_square[j]: cnt += 1
        j = hierarchy[0,j,1]
    return cnt

def get_contours(image, blur):
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur,blur), 0)
    edges   = cv2.Canny(blurred, 50, 150)
    dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    cnts, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    approx = [cv2.approxPolyDP(c, 0.04*cv2.arcLength(c,True),True) for c in cnts]
    return approx, hierarchy

def get_candidate_boards(cnts, hier, max_err, min_side, min_child):
    info  = [check_square(c) for c in cnts]
    is_sq = [(e<max_err and s>min_side) for e,s in info]
    boards = []
    for i, ((e,s),c) in enumerate(zip(info,cnts)):
        if is_sq[i] and child_count(i,hier,is_sq)>min_child:
            boards.append(c)
    return boards

def preprocess(image):
    all_boards = []
    for b in (3,5,7,9,11):
        cnts, hier = get_contours(image, b)
        all_boards += get_candidate_boards(cnts, hier, 4e2, 10, 10)
    scored = [(b, check_square(b)[1]) for b in all_boards]
    scored.sort(key=lambda x:-x[1])
    best = (None,0)
    for brd,side in scored:
        if side >= best[1]*0.9:
            best = (brd, side)
    return best[0]

def crop_image(image, pts):
    pts = np.squeeze(pts)
    x0,x1 = pts[:,0].min(), pts[:,0].max()
    y0,y1 = pts[:,1].min(), pts[:,1].max()
    return image[y0:y1, x0:x1]

# ── Custom layer & loss definitions ────────────────────────────────────────
class ChessboardLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        # strip any unexpected kwargs for Keras3
        kwargs.pop("dynamic", None)
        super().__init__(**kwargs)

    def call(self, inputs):
        # inputs: [batch,H,W,C] → output [batch,8,8,features]
        boards = tf.unstack(inputs, axis=0)
        out = []
        for b in boards:
            b = tf.expand_dims(b,0)
            rows = tf.split(b,8,axis=1)
            cells = []
            for r in rows:
                cs = tf.split(r,8,axis=2)
                flat = [tf.reshape(cc,(1,1,1,-1)) for cc in cs]
                cells.append(tf.concat(flat,axis=2))
            out.append(tf.concat(cells,axis=1))
        return tf.concat(out,axis=0)

    def compute_output_shape(self, input_shape):
        b,H,W,C = input_shape
        return (b,8,8,(H*W*C)//64)

def weighted_loss(y_true,y_pred):
    # identical to what model was trained with
    weights = tf.constant([1.0 if k==0 else 4.0 if k%6==0 else 8.0
                           for k in range(y_pred.shape[-1])], dtype=tf.float32)
    weights /= tf.reduce_max(weights)
    loss = keras.losses.sparse_categorical_crossentropy(y_true,y_pred)
    idx  = tf.reshape(y_true,(-1,))
    sw   = tf.reshape(tf.gather(weights,idx),loss.shape)
    return sw * loss

# ── Ensemble loading ──────────────────────────────────────────────────────
LOAD = [
    ("model/model_a.h5", 0.5),
    ("model/model_b.h5", 0.5),
]
_models = [
  keras.models.load_model(
    path,
    custom_objects={
      "ChessboardLayer": ChessboardLayer,
      "weighted_loss": weighted_loss
    },
    compile=False
  )
  for path,_ in LOAD
]

def ensemble_predict(x):
    acc = np.zeros((1,64,13),dtype=np.float32)
    for m,w in zip(_models,[w for _,w in LOAD]):
        acc += m.predict(x) * w
    return acc

# ── FastAPI endpoint ─────────────────────────────────────────────────────
@app.post("/scan")
async def scan_chessboard(file: UploadFile = File(...)):
    data = await file.read()
    arr  = np.frombuffer(data,dtype=np.uint8)
    img  = cv2.imdecode(arr,cv2.IMREAD_COLOR)

    pts     = preprocess(img)
    cropped = crop_image(img,pts)
    cropped = cv2.resize(cropped,(512,512),cv2.INTER_AREA)

    gray = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
    inp  = (gray.astype(np.float32)/255.0)[...,None][None,...]

    preds = ensemble_predict(inp)            # (1,64,13)
    grid  = np.argmax(preds,axis=-1).reshape(8,8)

    # map to chars + make FEN
    idx2ltr = {i:ch for i,ch in enumerate(
      ['p','n','b','r','q','k','P','N','B','R','Q','K','-']
    )}
    letter_grid = np.vectorize(lambda i: idx2ltr[int(i)])(grid)
    rows = []
    for rnk in letter_grid:
      cnt, rstr = 0, ""
      for c in rnk:
        if c=="-": cnt+=1
        else:
          if cnt: rstr+=str(cnt); cnt=0
          rstr+=c
      if cnt: rstr+=str(cnt)
      rows.append(rstr)
    fen = "/".join(rows) + " w - - 0 1"

    # overlay
    overlay = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    for i in range(8):
      for j in range(8):
        cv2.putText(overlay, letter_grid[i,j],
                    (j*64+5, i*64+50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0,255,255), 2)

    _,buf = cv2.imencode(".png",overlay)
    data_uri = "data:image/png;base64," + base64.b64encode(buf).decode()

    return {"fen": fen, "image": data_uri}
