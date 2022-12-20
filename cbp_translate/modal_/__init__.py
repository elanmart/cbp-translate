import os

# Explicit imports work better with IDEs
if os.getenv("MODAL_RUN_LOCALLY") == "1":
    from .local import SHARED, Container
    from .local import stub, volume, cpu_image, gpu_image
    from .local import hf_secret, deepl_secret, nemo_secret
else:
    from .remote import SHARED, Container
    from .remote import stub, volume, cpu_image, gpu_image
    from .remote import hf_secret, deepl_secret, nemo_secret
