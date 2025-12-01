import torch
import torch.nn as nn


class TCB_Kla(nn.Module):
    def __init__(
        self,
        d_model=128,             # Transformer Embedding维度
        num_heads=4,             # 多头注意力数
        max_len=45,              # 最大序列长度
        num_encoder_layers=2,    # Transformer层数
        conv_channels=128,        # 卷积通道数
        conv_kernels=(1, 3, 5),  # 卷积核尺寸
        dropout=0.5,             # dropout比例
        hidden_dim=128,           # GRU/LSTM隐藏维度
        fc_hidden=(1024, 512, 128)  # 全连接层维度
    ):
        super(TCB_Kla, self).__init__()

        self.save_hyperparams = dict(
            d_model=d_model,
            num_heads=num_heads,
            max_len=max_len,
            num_encoder_layers=num_encoder_layers,
            conv_channels=conv_channels,
            conv_kernels=conv_kernels,
            dropout=dropout,
            hidden_dim=hidden_dim,
            fc_hidden=fc_hidden
        )

        self.max_pool = 2
        self.embedding_size = d_model
        self.dropout_rate = dropout


        self.embeddingLayer = nn.Embedding(24, d_model, padding_idx=0)
        self.positionalEncodings = nn.Parameter(torch.rand(max_len, d_model), requires_grad=True)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=dropout
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers, encoder_norm)

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=d_model, out_channels=conv_channels, kernel_size=k, stride=1, padding="same"),
                nn.BatchNorm1d(conv_channels),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            ) for k in conv_kernels
        ])

        concat_channels = conv_channels * len(conv_kernels)
        self.bigru = nn.GRU(
            input_size=concat_channels,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.bilstm = nn.LSTM(
            input_size=concat_channels,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )


        fc_input_dim = hidden_dim * 2 * (max_len // 2)
        fc1, fc2, fc3 = fc_hidden

        self.line1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_dim, fc1),
            nn.BatchNorm1d(fc1),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(fc1, fc2),
            nn.BatchNorm1d(fc2),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(fc2, fc3),
        )

        self.line2 = nn.Sequential(
            nn.Linear(fc3, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
        )

        self.softmax_layer = nn.Softmax(dim=1)


    def forward(self, seq, pos):
        pad_mask = seq.eq(0)

        Embedding = self.embeddingLayer(seq)
        pos = pos.to(Embedding.dtype).to(Embedding.device)
        Embedding = Embedding + pos

        Embedding = Embedding.permute(1, 0, 2)
        feature = self.transformer_encoder(Embedding, src_key_padding_mask=pad_mask)
        feature = feature.permute(1, 0, 2).permute(0, 2, 1)

        conv_outs = [conv(feature) for conv in self.conv_blocks]
        feature_cat = torch.cat(conv_outs, dim=1)
        feature_cat = feature_cat.permute(0, 2, 1)


        lstm_out, _ = self.bilstm(feature_cat)
        output = self.line1(lstm_out)
        output = self.line2(output)
        out = self.softmax_layer(output)

        return out, output
