import os

checkpoint_path = os.environ.get("DNINET_CHECKPOINT_DIR", "./checkpoint").rstrip("/") + "/"
history_checkpoint_path = os.environ.get("DNINET_HISTORY_DIR", "./checkpoint/history").rstrip("/") + "/"
