import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from transformers import DistilBertModel

class MultiModalGarbageClassifier(nn.Module):
    """
    Multi-modal model for garbage classification.
    Uses MobileNetV2 for image processing and DistilBERT for text processing.
    """

    def __init__ (self, num_classes= 4, pretrained_cnn= True, pretrained_distilbert= True):
        """
        Parameters:
            num_classes (int): Number of classes in the classification task.
            pretrained_cnn (bool): If True, uses pre-trained MobileNetV2.
            pretrained_distilbert (bool): If True, uses pre-trained DistilBERT.
        """
        super(MultiModalGarbageClassifier, self).__init__()

        # [IMAGE FEATURE EXTRACTOR (MobileNetV2)]
        self.image_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

        # Replace the classifier layer to output a 256-dimensional feature vector
        num_features = self.image_model.classifier[1].in_features
        self.image_model.classifier[1] = nn.Linear(num_features, 256)

        # [TEXT FEATURE EXTRACTOR (DistilBERT)]
        # Since DistilBERT doesn't provide pooled output like BERT;
        # Manually extract the [CLS] "embedding" from last_hidden_state[:, 0, :].
        self.text_model = DistilBertModel.from_pretrained('distilbert-base-uncased') if pretrained_distilbert else DistilBertModel._from_config(...)

        # Reduce DistilBERT's hidden size (768) down to 256
        self.text_fc = nn.Linear(self.text_model.config.dim, 256)

        # Normalization layers for each modality
        self.img_norm = nn.LayerNorm(256)
        self.txt_norm = nn.LayerNorm(256)

        # [FUSION LAYER]
        # Fuse 256 (image) + 256 (text) = 512
        self.fusion_layer = nn.Linear(512, 512)

        # [CLASSIFICATION LAYER]
        self.classification_layer = nn.Linear(512, num_classes)

    def forward(self, images, text_input):
        """
        Forward pass of the multi-modal model.

        Parameters:
            images (Tensor): Image tensor of shape (batch_size, 3, 224, 224).
            text_inputs (Dict[str, Tensor]): Output of DistilBert tokenizer, e.g.:
            {
              'input_ids': Tensor(batch_size, max_length),
              'attention_mask': Tensor(batch_size, max_length)
            }
        Return:
            logits (Tensor): Output of the model before applying the softmax activation.
        """

        # [IMAGE FEATURE EXTRACTION]
        img_features = self.image_model(images)
        img_features = torch.relu(img_features) # shape => (batch_size, 256)
        img_features = self.img_norm(img_features)  # normalize

        # [TEXT FEATURE EXTRACTION]
        # DistilBERT returns a BaseModelOutput. The `last_hidden_state` is at index 0 or .last_hidden_state
        text_outputs = self.text_model(**text_input)
        # Extract [CLS] "token": DistilBERT uses the first token as the "classification token"
        # shape => (batch_size, sequence_length, hidden_size), we take [:, 0, :] for [CLS]
        last_hidden_state = text_outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :] # shape => (batch_size, 768) for distilbert-base-uncased

        # Pass through a linear layer to reduce dimension to 256
        text_features = self.text_fc(cls_token)
        text_features = torch.relu(text_features) # shape => (batch_size, 256)
        text_features = self.txt_norm(text_features)  # normalize

        # [FUSION LAYER]
        combined_features = torch.cat((img_features, text_features), dim=1) # shape => (batch_size, 512)
        fused_features = torch.relu(self.fusion_layer(combined_features))

        # [CLASSIFICATION LAYER]
        logits = self.classification_layer(fused_features) # shape => (batch_size, num_classes)

        return logits
    
# Quick test to check if the model works
if __name__ == "__main__":
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_input_ids = torch.randint(0, 30522, (batch_size, 16))  # random token IDs
    dummy_attention_mask = torch.ones((batch_size, 16))

    model = MultiModalGarbageClassifier(num_classes=4)
    outputs = model(
        dummy_images,
        {
            "input_ids": dummy_input_ids,
            "attention_mask": dummy_attention_mask
        }
    )
    print("Logits shape:", outputs.shape)  # (2, 4)