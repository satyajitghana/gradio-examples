import gradio as gr
from transformers import pipeline
import torch
from PIL import Image, ImageDraw
import numpy as np

CUSTOM_CSS = """
.output-panel {
    padding: 15px;
    border-radius: 8px;
    background-color: #f8f9fa;
}
"""

DESCRIPTION = """
# Zero-Shot Object Detection Demo

This demo uses OWL-ViT model to perform zero-shot object detection. You can:
- Upload an image or use your webcam
- Specify objects you want to detect (comma-separated)
- Adjust the confidence threshold
- Get real-time detection results

## Instructions
1. Upload an image or use webcam
2. Enter objects to detect (e.g., "person, car, dog, chair")
3. Adjust confidence threshold if needed
4. Click "Detect Objects" to process
"""

class ObjectDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector = pipeline(
            model="google/owlv2-base-patch16-ensemble",
            task="zero-shot-object-detection",
            device=self.device
        )
    
    def process_image(self, image, objects_to_detect, confidence_threshold=0.3, progress=gr.Progress()):
        if image is None or not objects_to_detect:
            return None
        
        progress(0.2, "Processing image...")
        
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Parse objects to detect
        candidate_labels = [obj.strip() for obj in objects_to_detect.split(",")]
        
        progress(0.5, "Detecting objects...")
        # Run detection
        predictions = self.detector(
            image,
            candidate_labels=candidate_labels,
            threshold=confidence_threshold
        )
        
        progress(0.8, "Drawing results...")
        # Draw predictions on image
        draw = ImageDraw.Draw(image)
        
        for prediction in predictions:
            box = prediction["box"]
            label = prediction["label"]
            score = prediction["score"]
            
            xmin, ymin, xmax, ymax = box.values()
            draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=2)
            draw.text((xmin, ymin-20), f"{label}: {score:.2f}", fill="red")
            
        progress(1.0, "Done!")
        return image

def create_demo():
    detector = ObjectDetector()
    
    with gr.Blocks(css=CUSTOM_CSS) as demo:
        gr.Markdown(DESCRIPTION)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Input Image",
                    type="pil",
                    sources=["upload", "webcam"]
                )
                
                objects_input = gr.Textbox(
                    label="Objects to Detect (comma-separated)",
                    placeholder="person, car, dog, chair",
                    value="person, face, phone, laptop"
                )
                
                confidence = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.3,
                    step=0.1
                )
                
                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary")
                    detect_btn = gr.Button("Detect Objects", variant="primary")
            
            with gr.Column(scale=1, elem_classes=["output-panel"]):
                output_image = gr.Image(
                    label="Detection Results",
                    type="pil"
                )
        
        # Event handlers
        detect_btn.click(
            fn=detector.process_image,
            inputs=[input_image, objects_input, confidence],
            outputs=[output_image]
        )
        
        clear_btn.click(
            fn=lambda: (None, None),
            inputs=[],
            outputs=[input_image, output_image]
        )
        
        # Add some example images
        # gr.Examples(
        #     examples=[
        #         ["examples/example1.jpg", "person, car, dog"],
        #         ["examples/example2.jpg", "laptop, phone, cup"],
        #     ],
        #     inputs=[input_image, objects_input],
        # )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    ) 