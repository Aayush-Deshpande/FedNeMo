"""FedNeMo: local, in-process federated fine-tuning of Nemotron-Mini-4B for
medical anomaly detection (ECG / cardiac risk).

No servers, no MCP, no cloud orchestration. The only outbound network call in the
entire system is a single request to NVIDIA's hosted nemotron-parse model at
inference time (document parsing of a scanned report image).
"""

__version__ = "0.1.0"
