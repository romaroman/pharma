"""
Dataset transformations visualization
"""

images = list()
for i in range(0, 10 ** 2):
    images.append((train_dataset.__getitem__(800)[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8))

image_combined = utils.combine_images(images)
cv.imwrite('example.png', image_combined)

