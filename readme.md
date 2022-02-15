# Public transport access detector

Object detection system inspired from YOLOv2.

Aim: provide a system that aids the visually impaired persons in using any public transport system, independent of any infrastructure by providing information about where the bus or car or about the registration plate of the vehicle (useful when using applications similar to Uber).

![cover](https://user-images.githubusercontent.com/46956225/149176877-5df91103-e5ee-493d-8fd6-93a8d74c38a2.jpg)

<a href="https://youtu.be/KleiULI0XbI">Early stages presentation</a>

<a href="https://github.com/ComanacDragos/PublicTransportDetector/tree/main/documentation/results/results_v10_5">v10_5 results</a>

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

![anchors_3](https://user-images.githubusercontent.com/46956225/154057057-793e7c63-0a98-485d-a85f-947b14c5e25c.png) ![anchors_4](https://user-images.githubusercontent.com/46956225/154057058-c2fc4be6-78b9-4ee7-8f8d-8885ef730261.png) ![anchors_5](https://user-images.githubusercontent.com/46956225/154057051-71f11858-42ac-4d0e-8ea5-2689cca02412.png) 
