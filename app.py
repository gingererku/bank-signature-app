import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

st.set_page_config(page_title="Bank Signature Verification", page_icon="✍️", layout="centered")

st.title("✍️ التحقق من التوقيع البنكي")
st.info("📱 **للاستخدام من الموبايل:** اضغطي على **Browse files** في كل خانة، واختاري **الكاميرا (Camera)** لتصوير الإمضاء مباشرة أو **المعرض**.")

# 1. معمارية الموديل
class SignatureNet(nn.Module):
    def __init__(self):
        super(SignatureNet, self).__init__()
        base_model = models.resnet18(weights=None)
        base_model.fc = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )
        self.backbone = base_model

    def forward(self, x):
        return F.normalize(self.backbone(x), p=2, dim=1)

@st.cache_resource
def load_model():
    model = SignatureNet()
    model.load_state_dict(torch.load("cedar_signature_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# 2. دالة معالجة صور الموبايل (تصحيح الدوران وتصغير الحجم وعزل الخلفية)
def preprocess_signature(pil_img):
    # تصحيح دوران الصورة التلقائي لكاميرا الموبايل (EXIF)
    pil_img = ImageOps.exif_transpose(pil_img)
    
    # تصغير الصورة لو جاية بدقة عالية من كاميرا الفون لتسريع المعالجة
    pil_img.thumbnail((800, 800))
    
    img = np.array(pil_img.convert("L"))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        thresh = thresh[y:y+h, x:x+w]
        
    return Image.fromarray(thresh).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_embedding(pil_img):
    processed = preprocess_signature(pil_img)
    tensor = transform(processed).unsqueeze(0)
    with torch.no_grad():
        emb = model(tensor)
    return emb, processed

# 3. واجهة رفع/تصوير سريعة للموبايل (بدون تعليق كاميرا المتصفح)
st.subheader("1️⃣ الإمضاءات المرجعية الثلاثة")
ref1 = st.file_uploader("📸 الإمضاء الأول", type=["png", "jpg", "jpeg"], key="r1")
ref2 = st.file_uploader("📸 الإمضاء الثاني", type=["png", "jpg", "jpeg"], key="r2")
ref3 = st.file_uploader("📸 الإمضاء الثالث", type=["png", "jpg", "jpeg"], key="r3")

st.divider()
st.subheader("2️⃣ الإمضاء الرابع (للاختبار)")
test_img_file = st.file_uploader("📸 الإمضاء الرابع", type=["png", "jpg", "jpeg"], key="test")

# عرض مصغرات سريعة للتأكد إن الصور اترفعت صح
cols = st.columns(4)
for idx, f in enumerate([ref1, ref2, ref3, test_img_file]):
    if f is not None:
        cols[idx].image(ImageOps.exif_transpose(Image.open(f)), caption=f"#{idx+1}", use_container_width=True)

# 4. زر التحقق
if st.button("🔍 تحقق من الإمضاء الآن", use_container_width=True, type="primary"):
    if ref1 and ref2 and ref3 and test_img_file:
        with st.spinner("جاري فحص التوقيع..."):
            imgs = [Image.open(f) for f in [ref1, ref2, ref3]]
            test_pil = Image.open(test_img_file)
            
            embs = [get_embedding(im)[0] for im in imgs]
            mean_ref_emb = F.normalize(torch.mean(torch.stack(embs), dim=0), p=2, dim=1)
            
            test_emb, processed_test = get_embedding(test_pil)
            
            similarity = F.cosine_similarity(mean_ref_emb, test_emb).item()
            accuracy_pct = max(0.0, min(100.0, similarity * 100))
            
        st.divider()
        st.metric(label="نسبة التطابق", value=f"{accuracy_pct:.2f}%")
        st.progress(int(accuracy_pct))
        
        if accuracy_pct >= 80.0:
            st.success("✅ **النتيجة: هو نفس الشخص** (التوقيع مطابق).")
        else:
            st.error("❌ **النتيجة: شخص مختلف / توقيع مزور**.")
            
        with st.expander("عرض الإمضاء بعد عزل الخلفية"):
            st.image(processed_test, width=220)
    else:
        st.warning("⚠️ يرجى تصوير أو اختيار الـ 4 إمضاءات أولاً.")