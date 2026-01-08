import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import av
import threading
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode, VideoTransformerBase

# ==========================================
# 1. CẤU HÌNH TRANG (Phải đặt đầu tiên)
# ==========================================
st.set_page_config(
    page_title="Pill Cam V22",
    layout="wide",
    page_icon="💊",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        #MainMenu, footer, header {visibility: hidden;}
        .block-container {padding: 0; max-width: 100%;}

        /* Video nền đen, bo góc */
        video {
            width: 100% !important; 
            border-radius: 8px; 
            background: #111;
        }

        /* Nút Chụp Ảnh To & Nổi bật */
        .stButton button[kind="primary"] {
            height: 65px;
            font-size: 20px; 
            font-weight: 900; 
            border-radius: 12px;
            background: #FF5252; /* Màu đỏ cam nổi bật */
            color: white;
            border: 2px solid white;
            width: 100%;
            margin-top: 10px;
        }

        /* Tinh chỉnh Menu Cuộn (Expander) */
        .streamlit-expanderHeader {
            font-weight: bold;
            background-color: #f0f2f6;
            border-radius: 8px;
            font-size: 16px;
        }

        div[data-testid="stTextInput"], div[data-testid="stNumberInput"] { margin-bottom: 0px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. KHỞI TẠO DỮ LIỆU (Tránh lỗi KeyError)
# ==========================================
if 'history' not in st.session_state: st.session_state['history'] = []
if 'last_snap' not in st.session_state: st.session_state['last_snap'] = None

# Khởi tạo giá trị mặc định cho Settings
defaults = {'target': 0, 'drug_name': "", 'zoom': 1.0, 'conf': 0.5}
for key, val in defaults.items():
    if key not in st.session_state: st.session_state[key] = val


# ==========================================
# 3. CORE AI
# ==========================================
@st.cache_resource
def load_model():
    return YOLO("best.pt")


try:
    model = load_model()
except Exception as e:
    st.error(f"Lỗi Model: {e}")
    st.stop()


# Đổi tên Class để Streamlit xóa Cache cũ
class PillProcessorV22(VideoTransformerBase):
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.last_frame = None
        self.model = model

        # Khởi tạo tham số an toàn
        self.conf = 0.5
        self.zoom = 1.0
        self.target = 0

        self.frame_count = 0
        self.skip_frames = 2  # Giúp giảm lag
        self.last_boxes = []
        self.last_count = 0

    def update_params(self, conf, zoom, target):
        self.conf = conf
        self.zoom = zoom
        self.target = target

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]

            # 1. ZOOM KỸ THUẬT SỐ
            if self.zoom > 1.0:
                nw, nh = int(w / self.zoom), int(h / self.zoom)
                x1, y1 = (w - nw) // 2, (h - nh) // 2
                cropped = img[y1:y1 + nh, x1:x1 + nw]
                img = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

            with self.frame_lock:
                self.last_frame = img.copy()

            # 2. AI (Skip Frame)
            self.frame_count += 1
            if self.frame_count % (self.skip_frames + 1) == 0:
                results = self.model(img, conf=self.conf)
                self.last_boxes = results[0].boxes
                self.last_count = len(self.last_boxes)

            # 3. VẼ KẾT QUẢ
            if self.last_boxes is not None:
                for box in self.last_boxes:
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    # Vẽ khung xanh lá
                    cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)

            # 4. HUD TRONG SUỐT (Header Bar)
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

            status = f"SL: {self.last_count}"
            color = (0, 255, 0)  # Xanh

            if self.target > 0:
                diff = self.last_count - self.target
                if diff == 0:
                    status += " (OK)"
                elif diff < 0:
                    status += f" (THIEU {abs(diff)})"
                    color = (0, 255, 255)  # Vàng
                else:
                    status += f" (DU {diff})"
                    color = (0, 0, 255)  # Đỏ

            # Text hiển thị
            cv2.putText(img, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except:
            return frame


# ==========================================
# 4. GIAO DIỆN CHÍNH
# ==========================================

col_cam, col_ctrl = st.columns([2, 1], gap="small")

# --- CỘT TRÁI: CAMERA ---
with col_cam:
    # Cấu hình STUN mạnh mẽ nhất của Google
    # Giúp xuyên tường lửa 4G tốt hơn bản cũ
    rtc_config = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]},
        ]
    })

    ctx = webrtc_streamer(
        key="pill-cam-v22-fix",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_processor_factory=PillProcessorV22,
        media_stream_constraints={
            "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}, "facingMode": "environment"},
            "audio": False
        },
        async_processing=True,
    )

# --- CỘT PHẢI: MENU DẠNG CUỘN ---
with col_ctrl:
    # MENU DẠNG CUỘN (EXPANDER)
    # Toàn bộ form nhập liệu nằm trong này cho gọn
    with st.expander("🛠️ CÀI ĐẶT & THÔNG SỐ (Bấm để mở)", expanded=False):

        # Dùng FORM để tránh reset camera khi đang gõ
        with st.form("settings_form"):
            st.write("📝 **Nhập thông tin:**")

            # Tên thuốc
            d_name = st.text_input("Tên thuốc/Lô", value=st.session_state['drug_name'])

            # Mục tiêu, Zoom, Độ nhạy
            c1, c2 = st.columns(2)
            with c1:
                t_val = st.number_input("🎯 Mục tiêu", min_value=0, value=st.session_state['target'])
            with c2:
                z_val = st.slider("🔍 Zoom", 1.0, 4.0, st.session_state['zoom'], 0.1)

            c_val = st.slider("🤖 Độ nhạy AI", 0.1, 1.0, st.session_state['conf'], 0.05)

            # Nút Áp dụng (Submit Form)
            if st.form_submit_button("✅ Áp dụng ngay", use_container_width=True):
                # Lưu vào session
                st.session_state['drug_name'] = d_name
                st.session_state['target'] = t_val
                st.session_state['zoom'] = z_val
                st.session_state['conf'] = c_val
                st.rerun()  # Refresh nhẹ để đẩy dữ liệu

    # Cập nhật thông số vào Camera (Real-time)
    if ctx.video_transformer:
        ctx.video_transformer.update_params(
            st.session_state['conf'],
            st.session_state['zoom'],
            st.session_state['target']
        )

    # NÚT CHỤP (Luôn hiển thị bên ngoài cho dễ bấm)
    st.write("")
    if st.button("📸 CHỤP ẢNH LƯU KẾT QUẢ", type="primary", use_container_width=True):
        if ctx.video_transformer and ctx.video_transformer.last_frame is not None:
            with ctx.video_transformer.frame_lock:
                frame_snap = ctx.video_transformer.last_frame.copy()

            # Xử lý ảnh tĩnh
            results = model(frame_snap, conf=st.session_state['conf'])
            cnt = len(results[0].boxes)

            # Vẽ lại
            for box in results[0].boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(frame_snap, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)

            final_snap = cv2.cvtColor(frame_snap, cv2.COLOR_BGR2RGB)
            st.session_state['last_snap'] = {'img': final_snap, 'count': cnt}

            # Ghi chú
            tgt = st.session_state['target']
            note = "---"
            if tgt > 0:
                if cnt == tgt:
                    note = "✅ ĐỦ"
                elif cnt < tgt:
                    note = f"⚠️ THIẾU {tgt - cnt}"
                else:
                    note = f"⛔ DƯ {cnt - tgt}"

            name_final = st.session_state['drug_name'] if st.session_state['drug_name'] else "---"

            ts = datetime.now().strftime("%H:%M:%S")
            st.session_state['history'].insert(0, {"Giờ": ts, "Tên": name_final, "SL": cnt, "Y/C": tgt, "Note": note})
            st.toast(f"Đã lưu: {cnt}", icon="💾")
        else:
            st.toast("⚠️ Hãy bật Camera trước!", icon="🚫")

    # PREVIEW (Xem ảnh vừa chụp)
    if st.session_state['last_snap'] is not None:
        snap = st.session_state['last_snap']
        st.image(snap['img'], caption=f"Kết quả: {snap['count']} viên", use_container_width=True)
        if st.button("Đóng Preview", use_container_width=True):
            st.session_state['last_snap'] = None
            st.rerun()

    # LỊCH SỬ & TẢI VỀ
    c_del, c_dl = st.columns(2)
    with c_del:
        if st.button("🗑️ Xóa hết"):
            st.session_state['history'] = []
            st.rerun()
    with c_dl:
        if st.session_state['history']:
            df = pd.DataFrame(st.session_state['history'])
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Excel", csv, "KiemKe.csv", "text/csv")

if st.session_state['history']:
    st.divider()
    st.dataframe(pd.DataFrame(st.session_state['history']), use_container_width=True, hide_index=True)