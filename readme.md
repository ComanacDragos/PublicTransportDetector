# Public transport object detector

Object detection system inspired from YOLOv2.

Aim: provide a system that aids the visually impaired persons in using any public transport system, independent of any infrastructure by providing information about busses or cars or about the registration plate of the vehicle (useful when using applications similar to Uber).

![cover](https://user-images.githubusercontent.com/46956225/169350678-1f402c18-87b7-4edb-ba9b-42f544f38fcf.png)

Bounding box color meanings:

Red: Bus

Green: Car

Blue: License plate
<hr>
<a href="https://youtu.be/KleiULI0XbI">Early stages presentation</a>

<a href="https://github.com/ComanacDragos/PublicTransportDetector/tree/main/object_detection/documentation/results/final_results_model_v42_fine_tuned">v42_fine_tuned results</a>

<a href="https://github.com/ComanacDragos/PublicTransportDetector/tree/main/object_detection/documentation/results/results_v10_5">v10_5 results</a>

Dataset used:  <a href="https://storage.googleapis.com/openimages/web/factsfigures_v4.html">Open Images Dataset v4</a>

Data augmentation:
<ul>
    <li>Mosaic</li>
    <li>Cutout</li>
    <li>Random hue</li>
    <li>Random brightness</li>
    <li>Random saturation</li>
    <li>Random contrast</li>
</ul>

Anchors generated using K-means

![anchors_3](https://user-images.githubusercontent.com/46956225/154057057-793e7c63-0a98-485d-a85f-947b14c5e25c.png)
