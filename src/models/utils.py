def dilation2padding(kernel_size, dilation):
    return (kernel_size * dilation - dilation) // 2
