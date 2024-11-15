import depth_pro
import gradio as gr
import matplotlib.cm as cm
import numpy as np
from depth_pro.depth_pro import DepthProConfig
from PIL import Image
import torch

# CSS for better styling - simplified
CUSTOM_CSS = """
.output-panel {
    padding: 15px;
    border-radius: 8px;
    background-color: #f8f9fa;
}
"""

DESCRIPTION = """
# Depth Pro: Sharp Monocular Metric Depth Estimation

This demo uses Apple's Depth Pro model to estimate depth from a single image. The model can:
- Generate high-quality depth maps
- Estimate focal length
- Process images in less than a second

## Instructions
1. Upload an image or use one of the example images
2. Click "Generate Depth Map" to process
3. View the depth map and estimated focal length
"""

class DepthEstimator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = DepthProConfig(
            patch_encoder_preset="dinov2l16_384",
            image_encoder_preset="dinov2l16_384",
            checkpoint_uri="./checkpoints/depth_pro.pt",
            decoder_features=256,
            use_fov_head=True,
            fov_encoder_preset="dinov2l16_384",
        )
        self.model, self.transform = depth_pro.create_model_and_transforms(config=self.config)
        self.model.eval()
        self.model.to(self.device)

    def process_image(self, input_image_path, progress=gr.Progress()):
        if input_image_path is None:
            return None, None

        progress(0.2, "Loading image...")
        image, _, f_px = depth_pro.load_rgb(input_image_path)
        
        progress(0.4, "Preprocessing...")
        image = self.transform(image)
        image = image.to(self.device)
        
        progress(0.6, "Generating depth map...")
        with torch.no_grad():
            prediction = self.model.infer(image, f_px=f_px)

        progress(0.8, "Post-processing...")
        depth_map = prediction["depth"].squeeze().cpu().numpy()
        focallength_px = prediction["focallength_px"]

        # Normalize and colorize depth map
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        colormap = cm.get_cmap("magma")  # Changed to magma for better visualization
        depth_map = colormap(depth_map)
        depth_map = (depth_map[:, :, :3] * 255).astype(np.uint8)
        depth_map = Image.fromarray(depth_map)

        progress(1.0, "Done!")
        return depth_map, float(focallength_px.item())

def create_demo():
    estimator = DepthEstimator()
    
    with gr.Blocks(css=CUSTOM_CSS) as demo:
        gr.Markdown(DESCRIPTION)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Input Image",
                    type="filepath",
                    sources=["upload", "webcam"]
                )
                
                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary")
                    submit_btn = gr.Button("Generate Depth Map", variant="primary")
                
            with gr.Column(scale=1, elem_classes=["output-panel"]):
                output_depth_map = gr.Image(
                    label="Depth Map",
                    show_label=True
                )
                output_focal_length = gr.Number(
                    label="Estimated Focal Length (pixels)",
                    precision=2
                )
        
        # Event handlers
        submit_btn.click(
            fn=estimator.process_image,
            inputs=[input_image],
            outputs=[output_depth_map, output_focal_length]
        )
        
        clear_btn.click(
            fn=lambda: (None, None, None),
            inputs=[],
            outputs=[input_image, output_depth_map, output_focal_length]
        )
        
        # Examples section commented out as in original
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    ) 