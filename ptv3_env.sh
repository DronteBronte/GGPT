pip install torch-sparse  # (add  `--no-build-isolation' if needed)
pip install torch-scatter # --no-build-isolation
pip install torch-geometric # --no-build-isolation
pip install torch-cluster # --no-build-isolation
cd Pointcept/ && pip install -e .
pip install spconv-cu121 # Ensure it matches your PyTorch CUDA version.
pip install flash-attn --no-build-isolation 
# Build flash-attn from source if needed. For example:
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install --no-build-isolation --no-cache-dir flash-attn
pip install gradio  # Required for the interactive demo app
pip install einops  # Required for model components

# Note that the recipe of Ptv3 installation's highly depends on your pytorch and cuda version. You need to change the version of spconv and flash-attention accordingly, or install them from source when needed.