import cv2
import torch
from torchvision import transforms
from model import build_model
import mediapipe as mp
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path=r"C:\Users\kirolosAnwar\Desktop\AMIT\models\sign_model_epoch3.pth"):
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint["classes"]
    model = build_model(len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    return model, classes

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # تم التصحيح هنا
])

def predict_image(img, model, classes):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transforms.ToPILImage()(img)
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)
    return classes[predicted.item()], conf.item()

def realtime_camera(model, classes):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(0)

    predictions_buffer = []
    buffer_size = 10  # نخزن آخر 10 فريمات

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        hand_detected = False
        final_label = "No Hand"
        avg_conf = 0.0

        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                x_min, y_min = w, h
                x_max, y_max = 0, 0

                # احسب البوكس (للمعالجة فقط بدون رسم)
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                # padding
                pad = 20
                x_min = max(0, x_min - pad)
                y_min = max(0, y_min - pad)
                x_max = min(w, x_max + pad)
                y_max = min(h, y_max + pad)

                # قص اليد
                hand_img = frame[y_min:y_max, x_min:x_max]
                if hand_img.size != 0:
                    hand_img = cv2.resize(hand_img, (200, 200))

                    label, conf = predict_image(hand_img, model, classes)

                    # خزّن التنبؤ
                    predictions_buffer.append((label, conf))
                    if len(predictions_buffer) > buffer_size:
                        predictions_buffer.pop(0)

                    # شوف أكتر label متكرر
                    labels_only = [p[0] for p in predictions_buffer]
                    final_label = max(set(labels_only), key=labels_only.count)

                    # متوسط الثقة لنفس الحرف
                    same_label_confs = [p[1] for p in predictions_buffer if p[0] == final_label]
                    avg_conf = sum(same_label_confs) / len(same_label_confs) if same_label_confs else 0

                    # اطبع في الكونسول
                    print(f"Prediction: {final_label} | Confidence: {avg_conf:.2f}")

        # عرض النتائج فقط بدون bounding box
        if hand_detected:
            # عرض التوقع في منتصف الشاشة
            text = f"Sign: {final_label} ({avg_conf*100:.1f}%)"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = 50
            
            # خلفية نص
            cv2.rectangle(frame, 
                         (text_x - 10, text_y - text_size[1] - 10),
                         (text_x + text_size[0] + 10, text_y + 10),
                         (0, 0, 0), -1)
            
            # النص
            cv2.putText(frame, text,
                       (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # رسالة عندما لا توجد يد
            text = "Show your hand sign"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = 50
            
            cv2.rectangle(frame, 
                         (text_x - 10, text_y - text_size[1] - 10),
                         (text_x + text_size[0] + 10, text_y + 10),
                         (0, 0, 0), -1)
            
            cv2.putText(frame, text,
                       (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Sign Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model, classes = load_model()
    print(f"عدد الكلاسات: {len(classes)} -> {classes}")
    realtime_camera(model, classes)