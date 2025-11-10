import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import time
import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import base64
from PIL import Image

# ------------------------------------------------
# Streamlit Page Config
# ------------------------------------------------
st.set_page_config(page_title="🛒 Smart Supermarket Billing", layout='wide')

# Background image
bg_path = "/supermarket/supermarket_bg.jpeg"
with open(bg_path, "rb") as f:
    bg_base64 = base64.b64encode(f.read()).decode()

st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("data:image/jpg;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        color: white;
    }}
    div.block-container {{
        background-color: rgba(0,0,0,0.55);
        border-radius: 12px;
        padding: 20px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------
# Load YOLO Model
# ------------------------------------------------
model = YOLO("yolov8n.pt")
model.to("cpu").eval()

# ------------------------------------------------
# Dummy Stocks, Prices, Weights
# ------------------------------------------------
if "stocks" not in st.session_state:
    st.session_state.stocks = {
        "apple": {"price": 120, "stock": 20, "weight": 0.2},
        "banana": {"price": 60, "stock": 15, "weight": 0.15},
        "milk": {"price": 45, "stock": 10, "weight": 1.0},
        "bread": {"price": 30, "stock": 12, "weight": 0.4},
    }

# ------------------------------------------------
# Session State Initialization
# ------------------------------------------------
if "billing_active" not in st.session_state:
    st.session_state.billing_active = False
if "cart" not in st.session_state:
    st.session_state.cart = []

# ------------------------------------------------
# Billing Functions
# ------------------------------------------------
def add_to_cart(label):
    if label in st.session_state.stocks and st.session_state.stocks[label]["stock"] > 0:
        st.session_state.stocks[label]["stock"] -= 1
        st.session_state.cart.append(label)

def calculate_bill():
    total = 0
    total_weight = 0
    items = []
    for item in st.session_state.cart:
        price = st.session_state.stocks[item]["price"]
        weight = st.session_state.stocks[item]["weight"]
        total += price
        total_weight += weight
        items.append([item, f"₹{price}", f"{weight} kg"])
    df = pd.DataFrame(items, columns=["Item", "Price", "Weight"])
    df.loc[len(df)] = ["TOTAL", f"₹{total}", f"{round(total_weight,2)} kg"]
    return df, total, total_weight

def generate_bill_pdf(df, total, total_weight):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    title = Paragraph("<b>Smart Supermarket Billing System</b>", styles["Title"])
    elements.append(title)
    elements.append(Spacer(1, 12))

    data = [df.columns.tolist()] + df.values.tolist()
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.white),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>Total Amount:</b> ₹{total}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Total Weight:</b> {round(total_weight,2)} kg", styles["Normal"]))
    elements.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ------------------------------------------------
# UI Layout
# ------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.title("🛒 Smart Billing System")

    start_btn = st.button("▶️ Start Billing", key="start_btn")
    stop_btn = st.button("🛑 Stop Billing", key="stop_btn")
    bill_btn = st.button("🧾 Generate Bill", key="bill_btn")

    if start_btn:
        st.session_state.billing_active = True
        st.session_state.cart = []
        st.success("✅ Billing Started")

    if stop_btn:
        st.session_state.billing_active = False
        st.warning("🛑 Billing Stopped")

with col2:
    st.header("📋 Cart Items")
    cart_placeholder = st.empty()

# Stock monitor
st.markdown("### 📦 Stock Monitor")
stock_placeholder = st.empty()

# ------------------------------------------------
# Video Stream
# ------------------------------------------------
blank_img = np.zeros((380, 640, 3), dtype=np.uint8)
frame_window = st.image(blank_img, channels="BGR")

cap = cv2.VideoCapture(0)
while st.session_state.billing_active and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()

    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        add_to_cart(label)

    frame_window.image(annotated, channels="BGR")

    # Update cart table
    cart_df, total, total_weight = calculate_bill()
    cart_placeholder.dataframe(cart_df, use_container_width=True)

    # Update stock monitor table
    stock_df = pd.DataFrame([
        [k, v["stock"], f"₹{v['price']}", f"{v['weight']} kg", "⚠️ Low" if v["stock"] <= 2 else "✅ OK"]
        for k, v in st.session_state.stocks.items()
    ], columns=["Item", "Stock Left", "Price", "Weight", "Status"])

    stock_placeholder.dataframe(stock_df, use_container_width=True)

    time.sleep(1)

cap.release()

# ------------------------------------------------
# Bill Display and PDF
# ------------------------------------------------
if bill_btn:
    bill_df, total, total_weight = calculate_bill()
    st.subheader("🧾 Final Bill Summary")
    st.dataframe(bill_df, use_container_width=True)

    pdf_buffer = generate_bill_pdf(bill_df, total, total_weight)
    pdf_base64 = base64.b64encode(pdf_buffer.read()).decode()

    st.download_button(
        label="📥 Download Bill PDF",
        data=pdf_buffer,
        file_name=f"bill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
    )

    st.markdown("### 🖨️ Bill Preview")
    pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="700" height="500" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)
