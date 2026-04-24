<div align="center">

# Retinal Fundus Feature Extraction Summary
**Mapped Deep Learning Latent Features for Glaucoma Detection**

</div>

This document outlines the specific physiological and morphological data inherently extracted and learned from the raw pixels of retinal fundus images by the deep learning architectures (YOLOv11, MambaOut, Vision Mamba).

While Deep Learning models act as "black boxes", visualization techniques (like Grad-CAM) and architectural analysis reveal that these networks mathematically isolate the following critical clinical biomarkers:

## 📊 Extracted Features Data Table

| 👁️ Feature / Data Point Extracted | 🩺 Clinical Relevance (Glaucoma Pathology) | 🤖 How the Deep Learning Model Captures It | ⚙️ Primary Extraction Layer |
| :--- | :--- | :--- | :--- |
| **Optic Cup-to-Disc Ratio (CDR)** | The primary geometric indicator of Glaucoma. A larger cup compared to the disc suggests retinal ganglion cell death. | The model identifies circular boundaries and contrast differentials between the bright optic cup and the darker surrounding disc. | Macro Spatial Encoders (Downsampling / Output Blocks) |
| **Neuroretinal Rim Thinning** | Deviations from the "ISNT" rule (Inferior ≥ Superior ≥ Nasal ≥ Temporal rim width) indicate glaucomatous damage. | Edge detection algorithms within the network learn the texture and thickness gradients bounding the optic disc. | Mid-Level Feature Map Convolutions |
| **Peripapillary Atrophy (PPA)** | Degeneration of the chorioretinal tissue immediately adjacent to the optic disc, highly correlated with disease probability. | Analyzed via sudden shifts in spatial frequency and irregular color pigmentation arrays adjacent to the central disc. | High-Frequency Texture Encoders |
| **Optic Disc Hemorrhages** | Splinter-like bleeding at the optic disc margin; an active marker of severe disease progression. | Extremely localized feature mapping identifying small, high-contrast red pixel clusters deviating from standard blood vessel morphology. | Early Edge/Color Convolutional Filters (Stem Layers) |
| **Blood Vessel "Bayoneting"** | Blood vessels taking a sharp right-angle turn as they cross the deep margin of a severely cupped optic disc. | Long-range spacial dependency mapping. *This is where State-Space Models (Mamba) excel over CNNs by tracking the continuous line of the vessel.* | Mamba SSM Scanning Sequences |
| **Retinal Nerve Fiber Layer (RNFL) Defects** | Wedge-shaped dark bands indicating loss of nerve fibers. Often the earliest sign of Glaucoma. | Low-contrast pattern recognition. The model maps the subtle luminance drops radiating outward from the optic disc. | Global Context Attention / Pooling Layers |
| **Illumination / Artifact Noise** | Reflection from the camera lens or uneven lighting from the fundus photographer. | Recognized as *negative* features. Thanks to our Color Jitter augmentation, the model actively learns to ignore these bright spots. | Regularization & Dropout Passes |

---

### Understanding the Extraction Difference: YOLO vs Mamba
- **YOLOv11 (CNN):** Extracts this data using sliding window matrices (convolutions). It is highly effective at finding localized features like Hemorrhages and Disc boundaries, but struggles slightly to piece together the whole image.
- **Mamba / Vision Mamba:** Extracts this data by flattening the image and analyzing it as a continuous sequence (linear scanning). It is vastly superior at tracking continuous data points, such as blood vessel bayoneting and wide RNFL defects spanning across the entire retina.
