import torch.nn as nn

class RoBERTaCNN(nn.Module):
    def __init__(self, roberta, num_classes, cnn_out_channels=100, cnn_kernel_size=2, cnn_stride=1):
        super(RoBERTaCNN, self).__init__()
        self.roberta = roberta
        self.embedding_dim = roberta.config.hidden_size
        self.cnn = nn.Conv1d(self.embedding_dim, cnn_out_channels, kernel_size=cnn_kernel_size, stride=cnn_stride)
        self.fc = nn.Linear(cnn_out_channels, num_classes)

    def forward(self, input_ids, attention_mask):
        # Get embeddings from RoBERTa
        embeddings = self.roberta(input_ids, attention_mask).last_hidden_state
        embeddings = embeddings.permute(0, 2, 1) # swap dimensions for CNN

        # Apply CNN
        cnn_features = self.cnn(embeddings)
        cnn_features = nn.functional.relu(cnn_features)
        cnn_features = nn.functional.max_pool1d(cnn_features, kernel_size=cnn_features.shape[-1])
        cnn_features = cnn_features.squeeze(dim=-1)

        # Apply connected layer
        output = self.fc(cnn_features)

        return output