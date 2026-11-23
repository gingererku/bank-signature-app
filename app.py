import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

# إعداد صفحة التطبيق لتكون متجاوبة مع شاشة الموبايل
st.set_page_config(page_title="Bank Signature Verification", page_icon="✍️", layout="centered")

st.title("✍️ نظام التحقق من التوقيع البنكي")
st.write("قم بمسح (Scan) أو رفع 3 إمضاءات مرجعية للعميل، ثم أدخل الإمضاء الرابع للتحقق.")

# 1. تعريف نفس معمارية الموديل
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

# 2. دالة معالجة صورة الكاميرا أو الملف المرفوع
def preprocess_signature(pil_img):
    img = np.array(pil_img.convert("L"))
    # تحسين التباين للتعامل مع إضاءة كاميرا الموبايل
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

# 3. واجهة إدخال الصور (كاميرا أو رفع ملف)
input_method = st.radio("اختر طريقة إدخال الإمضاءات:", ["📸 تصوير بالكاميرا (Scan)", "📁 رفع صور من الجهاز"], horizontal=True)

def get_image_input(label, key):
    if input_method == "📸 تصوير بالكاميرا (Scan)":
        return st.camera_input(label, key=key)
    else:
        return st.file_uploader(label, type=["png", "jpg", "jpeg"], key=key)

st.subheader("1️⃣ الإمضاءات المرجعية الثلاثة (للعميل)")
ref1 = get_image_input("الإمضاء المرجعي الأول", "r1")
ref2 = get_image_input("الإمضاء المرجعي الثاني", "r2")
ref3 = get_image_input("الإمضاء المرجعي الثالث", "r3")

st.divider()
st.subheader("2️⃣ الإمضاء الرابع (المراد اختباره)")
test_img_file = get_image_input("صورة الإمضاء الرابع للتحقق", "test")

# 4. زر التحقق وحساب الدقة
if st.button("🔍 تحقق من الإمضاء الآن", use_container_width=True, type="primary"):
    if ref1 and ref2 and ref3 and test_img_file:
        with st.spinner("جاري تحليل الخطوط والانحناءات..."):
            imgs = [Image.open(f) for f in [ref1, ref2, ref3]]
            test_pil = Image.open(test_img_file)
            
            # استخراج البصمات الثلاث وحساب المتوسط
            embs = [get_embedding(im)[0] for im in imgs]
            mean_ref_emb = F.normalize(torch.mean(torch.stack(embs), dim=0), p=2, dim=1)
            
            # استخراج بصمة الإمضاء الرابع
            test_emb, processed_test = get_embedding(test_pil)
            
            # حساب نسبة التطابق
            similarity = F.cosine_similarity(mean_ref_emb, test_emb).item()
            # تحويل القيمة من [-1, 1] أو [0, 1] إلى نسبة مئوية واضحة
            accuracy_pct = max(0.0, min(100.0, similarity * 100))
            
        st.divider()
        st.metric(label="نسبة التطابق (Accuracy / Similarity)", value=f"{accuracy_pct:.2f}%")
        st.progress(int(accuracy_pct))
        
        if accuracy_pct >= 80.0:
            st.success("✅ **النتيجة: هو نفس الشخص** (التوقيع مطابق ومعتمد).")
        else:
            st.error("❌ **النتيجة: شخص مختلف / توقيع مزور** (نسبة التطابق أقل من الحد المسموح).")
            
        with st.expander("عرض شكل الإمضاء بعد المعالجة (كما يراه الذكاء الاصطناعي)"):
            st.image(processed_test, caption="الإمضاء الرابع بعد عزل الخلفية", width=250)
    else:
        st.warning("⚠️ يرجى إدخال جميع الإمضاءات الأربعة أولاً.")