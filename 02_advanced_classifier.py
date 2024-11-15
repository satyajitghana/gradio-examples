import gradio as gr
import torch
import timm
from PIL import Image
import time
from tqdm import tqdm
import numpy as np

class ImageClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = timm.create_model('mambaout_base.in1k', pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
        
        import requests
        url = 'https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt'
        self.labels = requests.get(url).text.strip().split('\n')
    
    @torch.no_grad()
    def predict_batch(self, image_list, progress=gr.Progress(track_tqdm=True)):
        if isinstance(image_list, tuple) and len(image_list) == 1:
            image_list = [image_list[0]]
            
        if not image_list or image_list[0] is None:
            return [[{"none": 1.0}]]
            
        progress(0.1, desc="Starting preprocessing...")
        tensors = []
        
        # Process each image in the batch
        for image in image_list:
            if image is None:
                continue
            # Convert numpy array to PIL Image
            img = Image.fromarray(image).convert('RGB')
            tensor = self.transform(img)
            tensors.append(tensor)
            
        if not tensors:
            return [[{"none": 1.0}]]
            
        progress(0.4, desc="Batching tensors...")
        batch = torch.stack(tensors).to(self.device)
        
        progress(0.6, desc="Running inference...")
        outputs = self.model(batch)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        progress(0.8, desc="Processing results...")
        batch_results = []
        for probs in probabilities:
            top5_prob, top5_catid = torch.topk(probs, 5)
            result = {
                self.labels[idx.item()]: float(prob)
                for prob, idx in zip(top5_prob, top5_catid)
            }
            batch_results.append(result)
        
        progress(1.0, desc="Done!")
        # Return results in the required format: list of list of dicts
        return [batch_results]

# Create classifier instance
classifier = ImageClassifier()

# Create Gradio interface
demo = gr.Interface(
    fn=classifier.predict_batch,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=5),
    title="Advanced Image Classification with Mamba",
    description="Upload images for batch classification with the mambaout_base.in1k model",
    batch=True,
    max_batch_size=4
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860) 