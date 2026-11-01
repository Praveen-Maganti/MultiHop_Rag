import sys

# Prevent transformers from attempting to import incompatible torchvision nms bindings
if "torchvision" not in sys.modules:
    sys.modules["torchvision"] = None
  