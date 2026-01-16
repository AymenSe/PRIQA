import test


class Config:
    # ---------------------------
    # Reproducibility
    # ---------------------------
    seed = 42
    cuda_id = 2

    # ---------------------------
    # Paths
    # ---------------------------
    train_root = "/home/asekhri@sic.univ-poitiers.fr/ICIP_2026/fivek_dataset/train"
    test_root = "/home/asekhri@sic.univ-poitiers.fr/ICIP_2026/fivek_dataset/test"
    out_dir = "./SwinIR/experiments/swinir_jpeg"

    # ---------------------------
    # Training
    # ---------------------------
    epochs = 200
    batch_size = 6
    num_workers = 4

    lr = 2e-4
    weight_decay = 0.0
    grad_clip = 1.0
    log_interval = 100

    # ---------------------------
    # JPEG settings
    # ---------------------------
    qf_train = 10
    qf_test = 10
    h_size = 128

    # ---------------------------
    # Weights & Biases
    # ---------------------------
    wandb_project = "jpeg-artifact-removal"
    wandb_name = "swinir_dng_q10"
    wandb_mode = "online"      # "online" | "offline" | "disabled"
