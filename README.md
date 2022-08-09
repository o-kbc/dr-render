# dr-render


## Assets

### Models

### Textures

The images from the DTD were used to render the objects' textures and the images from the COCO/VOC2012 were used to render the skybox and ground plane textures.
To use the images for the textures and background, download the datasets from the links below and extract to the folder `textures/`

[Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
[Common Objects in Context (COCO)](https://cocodataset.org/#home)
[The Pascal Visual Object Classes (VOC2012)](http://host.robots.ox.ac.uk/pascal/VOC/)

Any desired image can be used as the texture for the distractors, skybox, and ground plane. If using a specific set of images (e.g., contextual background with specific objects), just download it and adjust the path on the textures variables (i.e., SKYBOX_TEXTURES, GROUND_TEXTURES, DIST_TEX).