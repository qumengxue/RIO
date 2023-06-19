# RIO Dataset
This repository is a demo of the RIO common set.

## Example
You can follow the steps in example.ipynb to read and visualize some sample ground truth.

Here is a ground truth example from the dataset: 

![example](./00006666.jpg)

The intention description is "you can use the thing to cut the food on the table".

## Data Format
The annotation contains: 

```
"height": height of the image,
"width": width of the image, 
"image_id": COCO image id,
"task_id": per-intention id, 
"expressions": intention description, e.g., "You can use the thing to cook meal."
"bbox_list": The bbox annotations of all instances of the class that can satisfy the intention.
"mask_list": The mask annotations of all instances of the class that can satisfy intention.
```

## Baseline Methods
The baselines in the paper include the following, please refer to the corresponding repositories.

[MDETR: Modulated Detection for End-to-End Multi-Modal Understanding](https://github.com/ashkamath/mdetr)

[TOIST: Task Oriented Instance Segmentation Transformer with Noun-Pronoun Distillation](https://github.com/AIR-DISCOVER/TOIST)

[SeqTR: A Simple yet Universal Network for Visual Grounding](https://github.com/sean-zhuh/SeqTR)

[PolyFormer: Referring Image Segmentation as Sequential Polygon Generation](https://github.com/amazon-science/polygon-transformer)