# Marine Zooplankton Detector

import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import os

# ==============================
# DEFINE CLASS COLORS
# ==============================
CLASS_COLORS = {
    "chaetognath": (255, 0, 0),       # Red
    "larval fish": (0, 255, 0),       # Green
    "Hydromedusa": (0, 0, 255),       # Blue
    "lobate ctenophore": (255, 255, 0), # Yellow
    "Pleurobrachia": (255, 0, 255),   # Magenta
    "shrimp": (0, 255, 255),          # Cyan
    "Siphonophore": (128, 0, 128),    # Purple
    "stomatopod larva": (0, 128, 128),# Teal
    "Unknown": (128, 128, 128),       # Gray
    "Thaliac": (255, 165, 0),         # Orange
    "polychaete worm": (75, 0, 130),  # Indigo
    "Cumacean": (238, 130, 238),      # Violet
    "ctenophore": (0, 100, 0),        # Dark Green
}

# ==============================
# LOAD YOUR TRAINED MODEL
# ==============================
def load_model():
    
    model_path = r"D:\Marine-AI\models\best.pt"
    
    print(f"Loading YOLOv8 model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return None, None

    try:
        model = YOLO(model_path)
        print("Model loaded successfully!")
        return model, model_path
    except Exception as e:
        print(f"Failed to load YOLO model: {str(e)}")
        return None, None

model, MODEL_PATH = load_model()


# ==============================
# DETECTION FUNCTION (CLEAN)
# ==============================
def detect_zooplankton(input_image):
    if input_image is None:
        return None, None, "<div style='text-align: center; color: #ff6b6b; font-size: 18px;'>Please upload an image first.</div>"
    
    if model is None:
        error_html = """
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #ff6b6b, #ee5a24); border-radius: 15px; color: white;'>
            <h3 style='margin: 0; font-size: 24px;'>Model Not Found</h3>
            <p style='margin: 10px 0 0 0; opacity: 0.9;'>Please ensure 'best.pt' is in your directory and restart the application.</p>
        </div>
        """
        return input_image, input_image, error_html
    
    img = np.array(input_image)
    original_img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    results = model.predict(source=img, conf=0.3, save=False, show=False)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    detections_by_class = defaultdict(int)
    total_confidence = defaultdict(list)
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls.item())
        confidence = float(box.conf.item())
        class_name = model.names[cls_id]
        
        detections_by_class[class_name] += 1
        total_confidence[class_name].append(confidence)
        
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 3)
        
        label = f"{class_name}"
        font_scale = 0.7
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img_rgb, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
        cv2.putText(img_rgb, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    if not detections_by_class:
        html_report = """
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 15px; color: white;'>
            <h3 style='margin: 0; font-size: 24px;'>No Marine Life Detected</h3>
            <p style='margin: 10px 0 0 0; opacity: 0.8;'>Try uploading a different image or adjusting the detection settings.</p>
        </div>
        """
    else:
        total_detections = sum(detections_by_class.values())
        sorted_detections = sorted(detections_by_class.items(), key=lambda x: x[1], reverse=True)
        
        color_legend = ""
        for class_name, count in sorted_detections:
            rgb_color = CLASS_COLORS.get(class_name, (128, 128, 128))
            hex_color = "#{:02x}{:02x}{:02x}".format(*rgb_color)
            avg_conf = np.mean(total_confidence[class_name]) * 100 if total_confidence[class_name] else 0
            
            color_legend += f"""
            <div style='display: flex; align-items: center; margin: 10px 0; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 10px; border-left: 5px solid {hex_color};'>
                <div style='flex: 1;'>
                    <div style='font-size: 18px; font-weight: bold; color: {hex_color};'>{class_name}</div>
                    <div style='font-size: 14px; opacity: 0.8;'>Count: {count} | Avg. Confidence: {avg_conf:.1f}%</div>
                </div>
                <div style='font-size: 24px; font-weight: bold; color: {hex_color};'>{count}</div>
            </div>
            """
        
        most_common = sorted_detections[0]
        diversity = len(detections_by_class)
        
        html_report = f"""
        <div style='background: linear-gradient(135deg, #667eea, #764ba2); padding: 25px; border-radius: 15px; color: white; font-family: Arial, sans-serif;'>
            <div style='text-align: center; margin-bottom: 25px;'>
                <h2 style='margin: 0; font-size: 28px;'>Marine Life Detection Results</h2>
                <p style='margin: 5px 0 0 0; opacity: 0.9; font-size: 16px;'>Analysis Complete • {total_detections} organisms detected</p>
            </div>
            
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 25px;'>
                <div style='text-align: center; padding: 20px; background: rgba(255,255,255,0.15); border-radius: 12px;'>
                    <div style='font-size: 24px; font-weight: bold;'>{total_detections}</div>
                    <div style='font-size: 14px; opacity: 0.8;'>Total Detections</div>
                </div>
                <div style='text-align: center; padding: 20px; background: rgba(255,255,255,0.15); border-radius: 12px;'>
                    <div style='font-size: 24px; font-weight: bold;'>{diversity}</div>
                    <div style='font-size: 14px; opacity: 0.8;'>Species Types</div>
                </div>
                <div style='text-align: center; padding: 20px; background: rgba(255,255,255,0.15); border-radius: 12px;'>
                    <div style='font-size: 16px; font-weight: bold;'>{most_common[0]}</div>
                    <div style='font-size: 14px; opacity: 0.8;'>Most Common</div>
                </div>
            </div>
            
            <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 12px;'>
                <h3 style='margin: 0 0 15px 0; font-size: 20px; text-align: center;'>Detailed Breakdown</h3>
                {color_legend}
            </div>
        </div>
        """
    
    return original_img, img_rgb, html_report

# ==============================
# GRADIO INTERFACE (CLEAN)
# ==============================
with gr.Blocks(
    title="Marine Zooplankton Detector",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan", neutral_hue="slate"),
    css="""
    .gradio-container { max-width: 1200px !important; margin: auto !important; }
    .upload-container { border: 2px dashed #3b82f6 !important; border-radius: 12px !important; background: linear-gradient(145deg, #f8fafc, #e2e8f0) !important; }
    .detection-button { background: linear-gradient(145deg, #3b82f6, #1d4ed8) !important; border: none !important; color: white !important; font-weight: bold !important; font-size: 16px !important; padding: 12px 24px !important; border-radius: 8px !important; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important; }
    """
) as demo:
    gr.Markdown("""
    # Marine AI
    ### Powered by AI • Real-time Analysis • Scientific Accuracy
    Upload underwater microscopic sample to detect and analyze marine microorganisms with state-of-the-art machine learning model.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload Your Image")
            input_image = gr.Image(label="Select Marine Sample Image", type="pil", height=400, elem_classes=["upload-container"])
            detect_btn = gr.Button("Analyze Microorganism sample", variant="primary", size="lg", elem_classes=["detection-button"])
            gr.Markdown("### Work Flow")
            gr.Markdown("""
            1. Upload the Microscopic sample
            2. Image Quality Assessment Check
            3. Detection and Classification of microorganisms
            4. Counting and Statistical Analysis
            5. View Annotated Results and Summary Report
            6. Download Results for Further Research
            """)
        with gr.Column(scale=2):
            gr.Markdown("### Detection Results")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Before Detection")
                    before_image = gr.Image(label="Original Image", interactive=False, height=300)
                with gr.Column():
                    gr.Markdown("#### After Detection")
                    after_image = gr.Image(label="Annotated Results", interactive=False, height=300)
            gr.Markdown("### Analysis Summary")
            output_report = gr.HTML(value="<div style='text-align: center; padding: 40px; color: #64748b;'>Upload an image to see detailed analysis results here...</div>")
    
    detect_btn.click(fn=detect_zooplankton, inputs=input_image, outputs=[before_image, after_image, output_report])
    
    gr.Markdown("""
    ---
    <p style='text-align: center; color: #64748b; font-size: 14px;'>
        Marine Zooplankton Detection System • Built with YOLOv8 & Gradio<br>
        For research and educational purposes • Accuracy may vary with image quality
    </p>
    """)

# ==============================
# LAUNCH APP
# ==============================
if __name__ == "__main__":
    import socket
    from contextlib import closing

    def find_free_port():
        try:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.bind(('127.0.0.1', 0))
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                return s.getsockname()[1]
        except socket.error as e:
            print(f"Error finding free port: {e}")
            return 7860

    try:
        free_port = find_free_port()
        print(f"Found free port: {free_port}")
        print("Starting Clean Gradio App...")
        demo.launch(server_name="127.0.0.1", server_port=free_port, share=True, show_api=False)
    except Exception as e:
        print(f"Error launching app: {e}")
        print("Retrying with default settings...")
        demo.launch(share=True)




        
