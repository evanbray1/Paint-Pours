# Paint Pour Digital Art Generator

<p align="center"><img width="652" height="1102" alt="image" src="https://github.com/user-attachments/assets/8257179e-fcb8-42fc-92e0-cfd89093e73f" /></p>



## Overview
This project seeks to digitally replicate the visual style of "paint pour" artwork, where an artist pours multiple colors of acrylic paint onto a canvas to create flowy, organic patterns. This package attempts to recreate this style of art by thinking of it as a carefully-crafted 3D surface and accompanying colormap, shown as a contour plot. The program generates filled-contour images using fractal/Perlin noise, custom segmented colormaps, and optional Voronoi-style cell overlays, resulting in images that resemble real paint pours. 

Nearly every aspect of each image can be tweaked with user-defined variables. When no values are specified, sensible values are randomly selected from intelligently-guessed bounds. The real thrill of this script lies in picking an image size and a palette of colors you find appealing, then letting the algorithms work their magic. In minutes, you'll have an assortment of one-of-a-kind digital art pieces perfect for use as a computer wallpaper, phone background, or large-format canvas print.

## Intended Audience
This tool is presented here for two categories of people. 
1. Anyone interested in creating wallpapers, backgrounds, or banner art for various forms of digital media. The code is written to be readable and accessible to true Python novices, making it a great starting point for beginners who want to experiment with generative art.
2. People who requested a code sample from the author for professional purposes. 

## Features
- **Easy to Use:** Easily create paint pour-style images from a single function call with very few required parameters 
- **Customizable Parameters:** Over a dozen variables can be tweaked manually for creative control, or left to be randomly chosen from sensible ranges.
- **Voronoi Cell Overlays:** Add cell-like structures to simulate effects seen in real acrylic pours. Although best visualized with the 'prominent_cells' argument, more features are coming to incorporate these cells structures at smaller scales in ways that more closely mimic real paint pours.
- **Prominent Cells Mode:** The `prominent_cells` option produces a fundamentally different style of image by manipulating variable ranges, resulting in large "cells" (as the paint pour artists call them) being prominently featured in the foreground. This style was popular enough during user-testing that it warranted its own dedicated preset.
- **Flexible Saving:** Save images to a specified directory or default-chosen one, already properly rendered in the desired resolution. Or simply display them interactively.
- **Intermediate plots for troubleshooting:** Optionally display detailed intermediate plots to illustrate the many steps of the image generation process. A selection of these plots are shown below. 

<img width="2012" height="1131" alt="image" src="https://github.com/user-attachments/assets/cd36aaa1-a248-4df1-bcf6-6b0165aef093" />


## Example Usage
Some truly minimal examples to get started with:

```python
import paint_pour_tools as pptools

pptools.generate_paint_pour_images(image_dimensions=[1920, 1080])
pptools.generate_paint_pour_images(image_dimensions=[1920, 1080],prominent_cells=True)

# This will create some images similar to the one you see at the top of this README.
pptools.generate_paint_pour_images(
    image_dimensions=[500,1000],
    num_images=5,
    display_final_image=True,
    save_image=True,
    show_intermediate_plots=False,
    cmap_name='custom',
    custom_cmap_colors=['#dadfdb','#a2544c', '#e4bda2', '#f18c6b', '#5c3c37', '#ce9896', '#7a291c', '#ce3d47'],
    num_levels=90,
    octave_powers=[1, 0.1, 0.0, 0.005],
    stretch_value=4,
    seed=None)
```

For more advanced usage, you can specify additional parameters to customize the output. See the docstrings for the `generate_paint_pour_image()` function.

## Getting Started
1. Install the required Python packages (see below).
2. Run `Paint Pouring Sandbox.py`.
3. Adjust parameters as desired to experiment with different styles.

## Installation
This project requires no special installation instructions, aside from several common modules like NumPy, Matplotlib, and Scipy. 

## Features I would like to add someday
- **Silicone oil "cells":** When silicone oil is added to the paint mixture in certain layers, it changers the surface tension properties of that layer, which results in that layer breaking when it gets thin enough, and allowing the lower layers to show through. For example, the below image has lots of these cells. This has proven a tricky feature to simulate.
  - ![paint pour with cells](https://github.com/user-attachments/assets/db4a4a1e-d9e6-44b5-8413-62084241b013)

- **Inter-layer blending:** Sometimes, the artist will purposefully, loosely mix two colors of paint so they are not quite homogenous _prior_ to pouring them onto the canvas. The image shown at the top of this README exhibits some of this. This can be done by craftily converting some portions of the segmented colormap to a continuous colormap. 
- **Like-color grouping:** This is exhibited in the image at the top of this page, where there are several shades of red and brown appearing in a horizontal band through the middle, and a band of pinks and whites just beneath it. Currently, segmented colormaps are produced by _completely_ randomly sampling the chosen base colormap. By incorporating some bias to this choice, or sampling multiple colors in batches from subsections of the base colormap, I believe this effect can be recreated digitally. 

## Contact
Questions, suggestions, or feedback? Contact the author:
**Evan Bray** — Bray.EvanP@gmail.com

## Image Gallery
A small, arbitrarily-chosen collection of images that this script is capable of producing. 

<img width="1965" height="1107" alt="image" src="https://github.com/user-attachments/assets/b68c49b7-bc41-401c-b857-113d5ed86162" />

