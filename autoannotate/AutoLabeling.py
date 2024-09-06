import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from segment_anything import sam_model_registry, SamPredictor
from pycocotools import mask as coco_mask


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def masks_to_rle(m_masks):
    rle_masks = []
    rle = coco_mask.encode(np.asfortranarray(m_masks[0]))  # Asegúrate de que la máscara esté en formato Fortran
    rle_masks.append(rle)
    return rle_masks


def create_coco_json(image_id, coco_masks, scores, image_info):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Información de la imagen
    coco_format["images"].append(image_info)

    # Añadir categorías (puedes personalizar esto según tus necesidades)
    coco_format["categories"].append({
        "id": 1,
        "name": "A",  # Cambia esto al nombre de tu clase
        "supercategory": "none"
    })

    # Añadir anotaciones
    for i, mask in enumerate(coco_masks):
        rle = coco_mask.encode(np.asfortranarray(mask))
        area = coco_mask.area(rle)[0]
        bbox = coco_mask.toBbox(rle).tolist()  # [x, y, width, height]

        annotation = {
            "id": i + 1,
            "image_id": image_id,
            "category_id": 1,  # Cambia esto al ID de tu categoría
            "segmentation": rle,
            "area": area,
            "bbox": bbox,
            "iscrowd": 0
        }
        coco_format["annotations"].append(annotation)

    return coco_format


image = cv2.imread(
    r'C:\Users\Esteban\OneDrive - Universidad EIA\Sistema de traduccion automatico\data\A\001\der\hand_detection\A001_42_r_hd.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('on')
plt.show()

sam_checkpoint = "checkpoint/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

predictor.set_image(image)

input_box = np.array([150, 100, 500, 500])
input_point = np.array([[300, 300], [250, 400], [350, 450], [170, 480]])
input_label = np.array([1, 1, 1, 0])

masks, scores, logits = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
)
mask_input = logits[np.argmax(scores), :, :]

masks, _, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    mask_input=mask_input[None, :, :],
    multimask_output=False,
)

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(masks[0], plt.gca())
show_points(input_point, input_label, plt.gca())
show_box(input_box, plt.gca())
plt.axis('on')
plt.show()

image_id = 1  # Cambia esto según tu lógica
image_info = {
    "id": image_id,
    "width": image.shape[1],
    "height": image.shape[0],
    "file_name": "A001_42_r_hd.jpg"  # Cambia esto al nombre de tu archivo
}

# Convertir las máscaras a RLE
rle_masks = masks_to_rle(masks)

# Crear el JSON en formato COCO
coco_json = create_coco_json(image_id, rle_masks, scores, image_info)

# Guardar el JSON en un archivo
with open('annotations.json', 'w') as json_file:
    json.dump(coco_json, json_file)
