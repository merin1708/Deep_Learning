from PIL import Image
from torchvision.transforms import functional as TF  

# 1. Load image
image = Image.open("DSC03861.JPG")

# 2. Adjust brightness (factor > 1 increases brightness)
bright_image = TF.adjust_brightness(image, 2)
contrast_image=TF.adjust_contrast(bright_image,1.5)
grayimage=TF.adjust_saturation(contrast_image,0)
gray=image.convert("L")
binary=gray.point(lambda p: 255 if p> 80 else 0)

bright_image.show()
contrast_image.show()
grayimage.show()
binary.show()
