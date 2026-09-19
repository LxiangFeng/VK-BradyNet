# Patch: add KeypointOnly model to video_model_builder.py
# Insert after KeypointTemporalEncoder definition (or after MViTFusion class).

@MODEL_REGISTRY.register()
class KeypointOnly(nn.Module):
    """
    Keypoint-only baseline for ablation study.

    Input:
        keypoints: [B,T,21,10] or [B,T,210]

    Output:
        logits: [B,num_classes]
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.keypoint_encoder = KeypointTemporalEncoder(cfg)

        self.head = nn.Sequential(
            nn.Dropout(float(cfg.MODEL.DROPOUT_RATE)),
            nn.Linear(
                self.keypoint_encoder.out_dim,
                cfg.MODEL.NUM_CLASSES
            )
        )

    def forward(
        self,
        x=None,
        keypoints=None,
        bboxes=None,
        return_attn=False
    ):

        if keypoints is None:
            raise ValueError(
                "KeypointOnly requires keypoints input."
            )

        kpt_feature = self.keypoint_encoder(keypoints)

        logits = self.head(kpt_feature)

        return logits
