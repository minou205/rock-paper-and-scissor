import base64,time,socket,threading,os,ssl
import numpy as np
import cv2
from pathlib import Path
from flask import Flask,render_template,request
from flask_socketio import SocketIO,emit,join_room,leave_room
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH="rps_model_v2.keras"
CLASSES=["rock","paper","scissors"]
IMG_SIZE=(224,224)
CONF_MIN=0.60
PORT=5443
CERT_FILE="cert.pem"
KEY_FILE="key.pem"

SKIN_LOWER=np.array([0,133,77],dtype=np.uint8)
SKIN_UPPER=np.array([255,173,127],dtype=np.uint8)
MIN_AREA=4000
_KERN=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))

app=Flask(__name__,template_folder="templates")
socketio=SocketIO(app,cors_allowed_origins="*",async_mode="threading",max_http_buffer_size=8*1024*1024)
print("⏳  Loading model …")
model = load_model(MODEL_PATH)
print(f"✅  {MODEL_PATH} ready\n")
rooms:dict[str, list[dict]]={}
_lock=threading.Lock()

def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8",80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def generate_cert(ip: str):
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import ipaddress, datetime
    except ImportError:
        print("❌  'cryptography' not installed.\n\tRun: pip install cryptography")
        raise SystemExit(1)

    print(f"🔐  Generating self-signed certificate for {ip} …")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, ip),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME,"RPS Game"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject).issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow()+datetime.timedelta(days=3650))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.IPAddress(ipaddress.IPv4Address(ip)),
                x509.DNSName("localhost"),
            ]), critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    Path(CERT_FILE).write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    Path(KEY_FILE).write_bytes(
        key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.TraditionalOpenSSL,serialization.NoEncryption()))
    print(f"✅  Certificate saved ({CERT_FILE} / {KEY_FILE})\n")


def cert_matches_ip(ip: str) -> bool:
    try:
        from cryptography import x509
        cert=x509.load_pem_x509_certificate(Path(CERT_FILE).read_bytes())
        san=cert.extensions.get_extension_for_class(
            x509.SubjectAlternativeName).value
        return ip in [str(i) for i in san.get_values_for_type(x509.IPAddress)]
    except Exception:
        return False


def decode_frame(b64: str):
    try:
        _, data = b64.split(",", 1)
        buf=np.frombuffer(base64.b64decode(data),np.uint8)
        return cv2.imdecode(buf,cv2.IMREAD_COLOR)
    except Exception:
        return None

def classify(frame: np.ndarray):
    H,W = frame.shape[:2]
    ycrcb=cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)
    mask=cv2.inRange(ycrcb, SKIN_LOWER, SKIN_UPPER)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,_KERN)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,_KERN)
    mask=cv2.dilate(mask,_KERN,iterations=1)
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    lg=max(cnts, key=cv2.contourArea)
    if cv2.contourArea(lg) < MIN_AREA:
        return None, 0.0
    x,y,bw,bh = cv2.boundingRect(lg)
    p= int(max(bw, bh) * .20)
    x1,y1= max(0,x-p), max(0,y-p)
    x2,y2= min(W,x+bw+p), min(H,y+bh+p)
    crop=frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None,0.0
    ch,cw = crop.shape[:2]
    side= max(ch, cw)
    canvas= np.zeros((side, side, 3), dtype=np.uint8)
    canvas[(side-ch)//2:(side-ch)//2+ch,(side-cw)//2:(side-cw)//2+cw]=crop
    rgb= cv2.cvtColor(cv2.resize(canvas, IMG_SIZE), cv2.COLOR_BGR2RGB)
    tensor= preprocess_input(rgb.astype("float32"))
    probs=model.predict(np.expand_dims(tensor, 0), verbose=0)[0]
    idx=int(np.argmax(probs))
    conf=float(probs[idx])
    return (CLASSES[idx] if conf >= CONF_MIN else None),conf


def encode_frame(frame: np.ndarray) -> str:
    small = cv2.resize(frame,(320,240))
    _, buf = cv2.imencode(".jpg",small,[cv2.IMWRITE_JPEG_QUALITY,55])
    return "data:image/jpeg;base64,"+base64.b64encode(buf).decode()


def decide(g1, g2) -> int:
    if g1 is None and g2 is None:return-1
    if g1 is None:return 2
    if g2 is None:return 1
    if g1 == g2:return 0
    return 1 if (g1, g2) in {("rock","scissors"),("scissors","paper"),("paper","rock")} else 2


def room_of(sid):
    with _lock:
        for rid, ps in rooms.items():
            if any(p["sid"] == sid for p in ps):
                return rid
    return None


def other(room_id, sid):
    with _lock:
        for p in rooms.get(room_id, []):
            if p["sid"] != sid:
                return p
    return None


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def on_connect():
    print(f"[+] {request.sid}")


@socketio.on("disconnect")
def on_disconnect():
    sid     = request.sid
    room_id = room_of(sid)
    if room_id:
        opp = other(room_id, sid)
        if opp:
            emit("opponent_left", {}, to=opp["sid"])
        with _lock:
            rooms[room_id] = [p for p in rooms[room_id] if p["sid"] != sid]
            if not rooms[room_id]:
                del rooms[room_id]
    print(f"[-] {sid}")


@socketio.on("join")
def on_join(data):
    username=(data.get("username") or "Player").strip()
    room_id=(data.get("room") or "").strip()
    if not room_id:
        emit("error",{"msg":"Room ID cannot be empty."}); return

    join_room(room_id)
    with _lock:
        if room_id not in rooms:
            rooms[room_id] = []
        if len(rooms[room_id]) >= 2:
            emit("error", {"msg": "Room is full."}); leave_room(room_id); return
        rooms[room_id].append({"sid": request.sid, "username": username})
        count= len(rooms[room_id])
        players=list(rooms[room_id])

    print(f"[room {room_id}] {username} ({count}/2)")

    if count == 1:
        emit("waiting", {"msg": "Waiting for opponent…"})
    else:
        p1, p2 = players
        emit("match_ready", {"opponent": p2["username"], "player_no": 1}, to=p1["sid"])
        emit("match_ready", {"opponent": p1["username"], "player_no": 2}, to=p2["sid"])
        print(f"[room {room_id}] ⚔  {p1['username']} vs {p2['username']}")


@socketio.on("stream_frame")
def on_frame(data):
    sid     = request.sid
    room_id = data.get("room")
    frame   = decode_frame(data.get("frame", ""))
    if frame is None:
        return
    gesture, conf = classify(frame)
    emit("ai_result", {"gesture": gesture, "conf": round(conf, 3)})
    opp = other(room_id, sid)
    if opp:
        emit("opponent_frame",
             {"frame": encode_frame(frame), "gesture": gesture, "conf": round(conf, 3)},
             to=opp["sid"])


@socketio.on("start_round")
def on_start_round(data):
    room_id = data.get("room")
    with _lock:
        ps = rooms.get(room_id, [])
    if len(ps) == 2:
        emit("begin_countdown", {"server_ts": time.time()}, to=room_id)
        print(f"[room {room_id}] 🕐 countdown")


@socketio.on("submit_gesture")
def on_submit(data):
    sid     = request.sid
    room_id = data.get("room")
    with _lock:
        if room_id not in rooms: return
        for p in rooms[room_id]:
            if p["sid"] == sid:
                p["gesture"] = data.get("gesture")
        players    = list(rooms[room_id])
        both_ready = all("gesture" in p for p in players)

    if both_ready:
        g1 = players[0].get("gesture")
        g2 = players[1].get("gesture")
        verdict = {
            "g1": g1, "g2": g2,
            "winner":  decide(g1, g2),
            "p1_name": players[0]["username"],
            "p2_name": players[1]["username"],
        }
        emit("verdict", verdict, to=room_id)
        with _lock:
            for p in rooms.get(room_id, []):
                p.pop("gesture", None)
        print(f"[room {room_id}] 🏆 {g1} vs {g2} → {verdict['winner']}")


# ── Startup ────────────────────────────────────────────────────────────────────

def print_qr(url: str):
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except ImportError:
        print("  (install qrcode for QR: pip install qrcode pillow)")


def announce(ip: str, port: int):
    url = f"https://{ip}:{port}"
    print("=" * 62)
    print("  🎮  RPS AI Game — HTTPS ready")
    print("=" * 62)
    print(f"  Your device  →  https://localhost:{port}")
    print(f"  Friend link  →  {url}")
    print()
    print("  ⚠️  One-time step on first visit:")
    print("     Chrome : click 'Advanced' → 'Proceed to … (unsafe)'")
    print("     Firefox: click 'Advanced' → 'Accept the Risk'")
    print("     After that, camera works normally.")
    print("=" * 62)
    print()
    print("  📷  Scan QR with friend's phone:")
    print()
    print_qr(url)
    print()


def start_mdns(ip: str, port: int):
    try:
        from zeroconf import Zeroconf, ServiceInfo
        import socket as _s
        info = ServiceInfo(
            "_https._tcp.local.",
            "rps-game._https._tcp.local.",
            addresses=[_s.inet_aton(ip)],
            port=port,
            server="rps-game.local.",
        )
        Zeroconf().register_service(info)
        print(f"📡  mDNS: https://rps-game.local:{port}")
    except Exception:
        pass


if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    local_ip = get_local_ip()

    if not (Path(CERT_FILE).exists() and Path(KEY_FILE).exists() and cert_matches_ip(local_ip)):
        generate_cert(local_ip)

    announce(local_ip, PORT)
    start_mdns(local_ip, PORT)

    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(CERT_FILE, KEY_FILE)

    socketio.run(app, host="0.0.0.0", port=PORT,
                 ssl_context=ssl_ctx, debug=False, use_reloader=False)
