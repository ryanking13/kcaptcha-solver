from PIL import Image, ImageFilter

def crop_img(img, sz=(96, 60)):
    center = (img.width // 2, img.height // 2)
    
    left = center[0] - sz[0] // 2
    right = center[0] + sz[0] // 2
    upper = center[1] - sz[1] // 2
    bottom = center[1] + sz[1] // 2

    return img.crop((left, upper, right, bottom))


def resize_img(img, sz=(96, 96), fill_color=255):
    resized_img = Image.new("LA", sz, fill_color)

    w, h = img.size
    start_w = (sz[0] - w) // 2
    start_h = (sz[1] - h) // 2
    return resized_img.paste(img, (start_w, start_h))


def to_grayscale(img):
    return img.convert("LA")


def filter_img(img):
    img = img.filter(ImageFilter.MeanFilter(3))
    return img.filter(ImageFilter.EDGE_ENHANCE)