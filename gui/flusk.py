from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

app = Flask(__name__)

# تحديد الجهاز
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# تحميل الموديل
def load_model(model_path=r"C:\Users\kirolosAnwar\Desktop\AMIT\models\sign_model_epoch2.pth"):
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint["classes"]

    model = models.densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    return model, classes

model, classes = load_model()

# نفس الـ transform بتاع التدريب
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def predict_image(image, model, classes):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]

# الصفحة الرئيسية
@app.route("/", methods=["GET", "POST"])
def index():
    final_text = None
    if request.method == "POST":
        uploaded_files = request.files.getlist("images")
        orders = request.form.getlist("order")

        # تأكد إن الصور ≤ 6
        if len(uploaded_files) > 6:
            final_text = "❌ مسموح ترفع 6 صور فقط!"
        else:
            # اربط الصور بالترتيب اللي دخله المستخدم
            files_with_order = list(zip(uploaded_files, map(int, orders)))
            files_with_order.sort(key=lambda x: x[1])  # ترتيب حسب order

            sentence = []
            for file, _ in files_with_order:
                if file and file.filename != "":
                    image = Image.open(file.stream).convert("RGB")
                    label = predict_image(image, model, classes)
                    sentence.append(label)

            final_text = " ".join(sentence)

    return render_template("index.html", result=final_text)

if __name__ == "__main__":
    app.run(debug=True)
