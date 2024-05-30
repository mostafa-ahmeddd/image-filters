from tkinter import *
import cv2
import numpy as np

root = Tk()

root.geometry("750x350")

bg = PhotoImage(file="pixels.png")
label1 = Label(root, image=bg)
label1.place(x=0, y=0, relheight=1, relwidth=1)


# frame1 = Frame(root)
# frame1.pack(pady = 200 )

def median():
    def get_median(arr):
        sorted_arr = np.sort(arr, axis=None)
        n = len(sorted_arr)
        if n % 2 == 0:
            return (sorted_arr[n // 2 - 1] + sorted_arr[n // 2]) / 2
        else:
            return sorted_arr[n // 2]

    def median_filter(img, kernel_size):
        height, width = img.shape[:2]
        pad = kernel_size // 2
        padded_img = np.pad(img, pad_width=pad, mode='constant')
        output_img = np.zeros((height, width), dtype=np.uint8)
        for i in range(pad, height + pad):
            for j in range(pad, width + pad):
                neighborhood = padded_img[i - pad:i + pad + 1, j - pad:j + pad + 1]
                median_value = get_median(neighborhood)
                output_img[i - pad, j - pad] = median_value
        return output_img

    image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

    filtered_image_np = median_filter(image, kernel_size=3)

    cv2.imshow('Original Image', image)
    cv2.imshow('Filtered Image', filtered_image_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def adaptive():
    def adaptive_max_filter(image, kernel_size):
        padded_image = cv2.copyMakeBorder(image, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2,
                                          cv2.BORDER_REPLICATE)
        filtered_image = np.zeros_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                roi = padded_image[i:i + kernel_size, j:j + kernel_size]
                adaptive_max = np.max(roi)
                # Update the filtered image
                filtered_image[i, j] = adaptive_max

        return filtered_image

    image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

    kernel_size = 3

    filtered_image = adaptive_max_filter(image, kernel_size)

    cv2.imshow('Original Image', image)
    cv2.imshow('Filtered Image', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def averaging():
    def average_filter(image, kernel_size):
        pad_size = kernel_size // 2
        padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT)

        output_image = np.zeros_like(image)

        for y in range(pad_size, padded_image.shape[0] - pad_size):
            for x in range(pad_size, padded_image.shape[1] - pad_size):
                roi = padded_image[y - pad_size:y + pad_size + 1, x - pad_size:x + pad_size + 1]
                average = np.mean(roi)
                output_image[y - pad_size, x - pad_size] = average

        return output_image

    image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

    kernel_size = 5

    filtered_image = average_filter(image, kernel_size)

    cv2.imshow('Original Image', image)
    cv2.imshow('Filtered Image', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gaussianF():
    def gaussian_filter(image, kernel_size=3, sigma=1.0):
        kernel = np.exp(-0.5 * (np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1) / sigma) ** 2)
        kernel = kernel / np.sum(kernel)

        padded_image = np.pad(image, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2)),
                              mode='constant')

        filtered_image = np.zeros_like(image, dtype=np.float64)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded_image[i:i + kernel_size, j:j + kernel_size]
                filtered_image[i, j] = np.sum(region * kernel)

        return filtered_image.astype(np.uint8)

    image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

    filtered_image = gaussian_filter(image, kernel_size=5, sigma=1.5)

    cv2.imshow('Original Image', image)
    cv2.imshow('Filtered Image', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def laplacian():
    image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Could not read the image.")
        exit()

    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])

    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant')

    laplacian_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            laplacian_image[i, j] = np.sum(padded_image[i:i + 3, j:j + 3] * laplacian_kernel)

    cv2.imshow('Original Image', image)
    cv2.imshow('Laplacian Filtered Image', laplacian_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def unsharp():
    def get_median(arr):
        sorted_arr = np.sort(arr, axis=None)
        n = len(sorted_arr)
        if n % 2 == 0:
            return (sorted_arr[n // 2 - 1] + sorted_arr[n // 2]) / 2
        else:
            return sorted_arr[n // 2]

    def median_filter(img, kernel_size):
        height, width = img.shape[:2]
        pad = kernel_size // 2
        padded_img = np.pad(img, pad_width=pad, mode='constant')
        output_img = np.zeros((height, width), dtype=np.uint8)
        for i in range(pad, height + pad):
            for j in range(pad, width + pad):
                neighborhood = padded_img[i - pad:i + pad + 1, j - pad:j + pad + 1]
                median_value = get_median(neighborhood)
                output_img[i - pad, j - pad] = median_value
        return output_img

    image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

    blurred_img = median_filter(image, kernel_size=3)
    k_value = 1
    mask = image - blurred_img
    sharpened_img = image + k_value * mask
    cv2.imshow('filtered Image', sharpened_img)


def highboost():
    def get_median(arr):
        sorted_arr = np.sort(arr, axis=None)
        n = len(sorted_arr)
        if n % 2 == 0:
            return (sorted_arr[n // 2 - 1] + sorted_arr[n // 2]) / 2
        else:
            return sorted_arr[n // 2]

    def median_filter(img, kernel_size):
        height, width = img.shape[:2]
        pad = kernel_size // 2
        padded_img = np.pad(img, pad_width=pad, mode='constant')
        output_img = np.zeros((height, width), dtype=np.uint8)
        for i in range(pad, height + pad):
            for j in range(pad, width + pad):
                neighborhood = padded_img[i - pad:i + pad + 1, j - pad:j + pad + 1]
                median_value = get_median(neighborhood)
                output_img[i - pad, j - pad] = median_value
        return output_img


    image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

    blurred_img = median_filter(image, kernel_size=3)
    k_value = 2
    mask = image - blurred_img
    sharpened_img = image + k_value * mask
    cv2.imshow('filtered Image', sharpened_img)


def robert():
    def roberts_cross_operator(image):
        kernel_x = np.array([[1, 0],
                             [0, -1]])

        kernel_y = np.array([[0, 1],
                             [-1, 0]])

        if len(image.shape) > 2:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        padded_image = np.pad(gray_image, pad_width=1, mode='constant', constant_values=0)

        gradient_x = np.zeros_like(gray_image)
        gradient_y = np.zeros_like(gray_image)
        gradient_magnitude = np.zeros_like(gray_image)

        for y in range(gray_image.shape[0]):
            for x in range(gray_image.shape[1]):
                gradient_x[y, x] = np.sum(kernel_x * padded_image[y:y + 2, x:x + 2])
                gradient_y[y, x] = np.sum(kernel_y * padded_image[y:y + 2, x:x + 2])

                gradient_magnitude[y, x] = np.sqrt(gradient_x[y, x] ** 2 + gradient_y[y, x] ** 2)

        return gradient_x, gradient_y, gradient_magnitude

    image = cv2.imread('image.jpg')

    gradient_x, gradient_y, gradient_magnitude = roberts_cross_operator(image)

    cv2.imshow('Original Image', image)
    cv2.imshow('Gradient X', gradient_x)
    cv2.imshow('Gradient Y', gradient_y)
    cv2.imshow('Gradient Magnitude', gradient_magnitude)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sobel():
    mylabel = Label(root, text="look")
    mylabel.pack()


def impulse():
    def add_impulse_noise(image, salt_prob, pepper_prob):
        noisy_image = np.copy(image)
        salt_pixels = np.random.rand(*image.shape) < salt_prob
        pepper_pixels = np.random.rand(*image.shape) < pepper_prob

        noisy_image[salt_pixels] = 255

        noisy_image[pepper_pixels] = 0

        return noisy_image

    # Load an image
    image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

    salt_probability = 0.01
    pepper_probability = 0.01

    noisy_image = add_impulse_noise(image, salt_probability, pepper_probability)

    cv2.imshow('Original Image', image)
    cv2.imshow('Noisy Image', noisy_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gaussian_noise():
    def add_gaussian_noise(image, mean=0, std_dev=25):
        noise = np.random.normal(mean, std_dev, image.shape)
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy_image

    image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

    noisy_image = add_gaussian_noise(image)

    cv2.imshow('Original Image', image)
    cv2.imshow('Noisy Image', noisy_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def uniform():
    def add_uniform_noise(image, intensity):
        noise = np.random.uniform(-intensity, intensity, image.shape).astype(np.float32)

        noisy_image = image.astype(np.float32) + noise

        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        return noisy_image

    image_path = 'image.jpg'
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to load image from path:", image_path)
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        noisy_image = add_uniform_noise(gray_image, intensity=50)

        cv2.imshow('Original Image', gray_image)
        cv2.imshow('Noisy Image', noisy_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def histogram_e():
    import cv2
    import numpy as np

    def calculate_histogram(img):
        histogram = np.zeros(256, dtype=np.uint32)

        for pixel_value in img.flatten():
            histogram[pixel_value] += 1

        return histogram

    def calculate_max(arr):
        max_value = arr[0]

        for value in arr:
            if value > max_value:
                max_value = value

        return max_value

    def calculate_cdf(hist):
        cdf = np.zeros_like(hist)

        cdf[0] = hist[0]
        for i in range(1, len(hist)):
            cdf[i] = cdf[i - 1] + hist[i]

        return cdf

    def histogram_equalization(img):
        hist = calculate_histogram(img)

        max_hist_value = calculate_max(hist)

        cdf = calculate_cdf(hist)

        cdf_normalized = cdf * max_hist_value / calculate_max(cdf)

        equalized_img = np.interp(img.flatten(), np.arange(256), cdf_normalized).reshape(img.shape)

        return equalized_img.astype(np.uint8)

    image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

    equalized_image = histogram_equalization(image)

    cv2.imshow('Original Image', image)
    cv2.imshow('Equalized Image', equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def histogram_s():
    mylabel = Label(root, text="look")
    mylabel.pack()


def compress():
    class Node:
        def __init__(self, freq, symbol):
            self.freq = freq
            self.symbol = symbol
            self.left = None
            self.right = None

        def __lt__(self, other):
            return self.freq < other.freq

    def generate_huffman_tree(freq_dict):
        nodes = [Node(freq, symbol) for symbol, freq in freq_dict.items()]
        while len(nodes) > 1:
            nodes.sort(key=lambda x: x.freq)
            left = nodes.pop(0)
            right = nodes.pop(0)
            merged = Node(left.freq + right.freq, None)
            merged.left = left
            merged.right = right
            nodes.append(merged)
        return nodes[0]

    def generate_huffman_codes(root, code="", codes={}):
        if root is not None:
            if root.symbol is not None:
                codes[root.symbol] = code
            generate_huffman_codes(root.left, code + "0", codes)
            generate_huffman_codes(root.right, code + "1", codes)
        return codes

    def compress_image(image):
        flat_image = image.flatten()
        freq_dict = {}
        for pixel in flat_image:
            if pixel in freq_dict:
                freq_dict[pixel] += 1
            else:
                freq_dict[pixel] = 1
        root = generate_huffman_tree(freq_dict)
        codes = generate_huffman_codes(root)
        compressed_data = "".join([codes[pixel] for pixel in flat_image])
        return compressed_data, codes

    def decompress_image(compressed_data, codes, shape):
        decoded_data = []
        reverse_codes = {v: k for k, v in codes.items()}
        code = ""
        for bit in compressed_data:
            code += bit
            if code in reverse_codes:
                decoded_data.append(reverse_codes[code])
                code = ""
        decoded_data = np.array(decoded_data, dtype=np.uint8)
        decompressed_image = np.reshape(decoded_data, shape)
        return decompressed_image

        # Load grayscale image

    image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

    # Compress image
    compressed_data, codes = compress_image(image)

    # Decompress image
    decompressed_image = decompress_image(compressed_data, codes, image.shape)

    # Display original and decompressed images
    cv2.imshow('Original Image', image)
    cv2.imshow('Decompressed Image', decompressed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def interpolation():
    def nearest_neighbor_interpolation(image, new_size):
        height, width = image.shape[:2]
        new_height, new_width = new_size

        scale_x = new_width / width
        scale_y = new_height / height

        new_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)

        for y in range(new_height):
            for x in range(new_width):
                src_x = min(int(x / scale_x), width - 1)
                src_y = min(int(y / scale_y), height - 1)
                new_image[y, x] = image[src_y, src_x]

        return new_image

    # Example usage
    image = cv2.imread("image.jpg")
    new_height = 300
    new_width = 400
    resized_image = nearest_neighbor_interpolation(image, (new_height, new_width))
    cv2.imshow("Resized Image", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


medianbutton = Button(root, text="median filter", command=median, fg="blue")
medianbutton.pack()
medianbutton.place(x=50, y=20)

adaptivebutton = Button(root, text="adaptive filter", command=adaptive, fg="blue")
adaptivebutton.pack()
adaptivebutton.place(x=150, y=20)

averagingbutton = Button(root, text="averaging filter", command=averaging, fg="blue")
averagingbutton.pack()
averagingbutton.place(x=250, y=20)

gaussianbutton = Button(root, text="gaussian filter", command=gaussianF, fg="blue")
gaussianbutton.pack()
gaussianbutton.place(x=350, y=20)

laplacianbutton = Button(root, text="laplacian filter", command=laplacian, fg="blue")
laplacianbutton.pack()
laplacianbutton.place(x=450, y=20)

unsharpbutton = Button(root, text="unsharp filter", command=unsharp, fg="blue")
unsharpbutton.pack()
unsharpbutton.place(x=550, y=20)

highboostbutton = Button(root, text="highboost filter", command=highboost, fg="blue")
highboostbutton.pack()
highboostbutton.place(x=650, y=20)

roberts_cross_gradient_button = Button(root, text="roberts_cross_gradient filter", command=robert, fg="blue", width=25)
roberts_cross_gradient_button.pack()
roberts_cross_gradient_button.place(x=50, y=100)

impulsenoisebutton = Button(root, text="impulse noise", command=impulse, fg="blue")
impulsenoisebutton.pack()
impulsenoisebutton.place(x=450, y=100)

gaussiannoisebutton = Button(root, text="gaussian noise", command=gaussian_noise, fg="blue")
gaussiannoisebutton.pack()
gaussiannoisebutton.place(x=550, y=100)

uniformbutton = Button(root, text="uniform noise", command=uniform, fg="blue")
uniformbutton.pack()
uniformbutton.place(x=650, y=100)

histoequalizationbutton = Button(root, text="histogram equalization", command=histogram_e, fg="blue", width=25)
histoequalizationbutton.pack()
histoequalizationbutton.place(x=450, y=200)

histospecializationbutton = Button(root, text="histogram specialization", command=histogram_s, fg="blue", width=25)
histospecializationbutton.pack()
histospecializationbutton.place(x=50, y=200)

compressbutton = Button(root, text="compression", command=compress, fg="blue")
compressbutton.pack()
compressbutton.place(x=250, y=200)

interpolationbutton = Button(root, text="interpolation ", command=interpolation, fg="blue")
interpolationbutton.pack()
interpolationbutton.place(x=360, y=200)

root.mainloop()
