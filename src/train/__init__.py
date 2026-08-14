"""LoRA fine-tuning pipeline for Chest-PRESS.

Three individually-triggerable stages (no merging — adapters are kept as adapters):
  qa         (train_qba.py)     — VQA grounding on MIMIC-Ext-CXR-QBA A++ subset → qa_adapter/
  reports    (train_reports.py) — CXR report generation on a fresh adapter      → reports_adapter/
  qa+reports (train_reports.py --init-adapter qa_adapter) — continue the qa adapter on the
             report data (PEFT continuation, is_trainable=True)                 → qa_reports_adapter/

The qa+reports stage requires the qa stage to be done first.  Training keeps adapters as
adapters (no merge); for INFERENCE the adapter is merged into the base (src/merge_adapter.py)
and served as a full checkpoint (src/inference.py --variant) — uniform across all architectures.
"""
