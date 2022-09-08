# off-axis-holography-python

## Description

Simple example of holographic reconstruction of an image aquired via off-axis-holography and also how to propagate the field


## Dependencies
numpy
matplotlib
scikit-image
scipy


## Examples
### Holographic reconstruction

Start from a holographic image and go to the full optical field.

<img src="samplefolder/PS_beads_1_1_1.png" width="482" height="362" title="Holographic image">

🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳🠳


<img src="plots/psl_111/phase_corrections2.png" title="1. Phase image pre unwrap, 2. Phase image post unwrap 3. Final phase image">

### Holographic reconstruction

From the optical field one can also propagate the field to any z.


<img src="plots/psl_111_propagate/z_propagation_real.png" width="482" height="482" title="Propagation to different z-planes">


