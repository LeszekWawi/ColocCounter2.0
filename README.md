# Fluorescence Microscopy Colocalization Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)]()

Advanced desktop application for comprehensive colocalization analysis in dual-channel fluorescence microscopy images, with specialized support for granule/condensate detection and quantification.

![Application Screenshot](docs/screenshot_placeholder.png)

## 🔬 Overview

This application provides a sophisticated, three-level analysis pipeline for studying protein colocalization in fluorescence microscopy:

1. **Global Pixel-Level Analysis** - Whole-cell intensity correlation
2. **Granule-Level Analysis** - Structure-based colocalization metrics
3. **Cross-Structure Analysis** - Bidirectional recruitment and enrichment quantification

### Key Features

- ✨ **Dual-Channel Analysis** - Simultaneous GFP and mCherry channel processing
- 🎯 **Granule Detection** - Automated detection of cellular structures/condensates
- 📊 **Multiple Metrics** - ICQ, Manders, CCS, Jaccard, enrichment ratios, recruitment ICQ
- 🔄 **Batch Processing** - Analyze multiple images with progress tracking
- 📈 **Real-time Visualization** - Interactive parameter tuning and live preview
- 💾 **Comprehensive Export** - CSV, HTML reports with detailed statistics
- 🖥️ **User-Friendly GUI** - Intuitive Tkinter-based interface

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Features](#-features)
- [Analysis Methods](#-analysis-methods)
- [Usage Guide](#-usage-guide)
- [Output Metrics](#-output-metrics)
- [Requirements](#-requirements)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Option 1: Clone Repository (Recommended)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/fluorescence-colocalization-analyzer.git
cd fluorescence-colocalization-analyzer

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Direct Download

1. Download ZIP from GitHub
2. Extract to your desired location
3. Open terminal in extracted folder
4. Run: `pip install -r requirements.txt`

### Dependencies

The application requires the following Python packages:

```
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
Pillow>=8.3.0
```

All dependencies are automatically installed via `requirements.txt`.

## ⚡ Quick Start

```bash
# Run the application
python export_problem.py
```

### Basic Workflow

1. **Select Folder** - Click "Browse" and select folder containing your .tif/.tiff images
2. **Choose Analysis Mode** - Select "GFP Granule Analysis" or "mCherry Granule Analysis"
3. **Adjust Parameters** (optional) - Fine-tune detection sensitivity in Parameters tab
4. **Process Images** - Click "Process Batch" button
5. **View Results** - Explore different visualization tabs
6. **Export Data** - Save results as CSV for further analysis

## ✨ Features

### Multi-Level Analysis Pipeline

#### Level 1: Global Pixel Analysis
- **Manders Coefficients (M1, M2)** - Fraction of each channel that colocalizes
- **Overlap Coefficient** - Overall colocalization strength
- **Whole-Cell ICQ** - Intensity correlation across entire image

#### Level 2: Granule-Level Analysis
- **Automated Granule Detection** - LoG (Laplacian of Gaussian) based segmentation
- **Granule ICQ** - Correlation within detected structures
- **ICQ Enhancement** - Comparison of ICQ in structures vs whole cell

#### Level 3: Cross-Structure Analysis
- **Physical Overlap** - Jaccard and Dice coefficients for structure overlap
- **Bidirectional CCS** - Conditional colocalization scores in both directions
- **Recruitment ICQ** - Directional recruitment analysis (protein A → B vs B → A)
- **Enrichment Ratios** - Quantify fold-enrichment in structures
- **Asymmetry Metrics** - Identify dominant recruitment direction

### Analysis Modes

#### GFP Granule Analysis Mode
- Detects GFP-positive structures
- Analyzes mCherry signal relative to GFP granules
- Answers: "Does mCherry recruit to GFP structures?"

#### mCherry Granule Analysis Mode
- Detects mCherry-positive structures
- Analyzes GFP signal relative to mCherry granules
- Answers: "Does GFP recruit to mCherry structures?"

### Visualization Options

- 🎨 **All Channels View** - Side-by-side GFP, mCherry, and RGB overlay
- 🟡 **Intensity-Based Colocalization** - Otsu threshold-based overlap
- 🌐 **Whole-Cell ICQ** - Global correlation heatmap
- 🔬 **Granule-Level ICQ** - Structure-specific correlation
- 📊 **Physical Overlap** - Venn diagram and overlap metrics
- 📈 **Enrichment Analysis** - Fold-enrichment visualization
- 🎯 **Recruitment ICQ** - Bidirectional recruitment comparison
- 📋 **Method Comparison** - Side-by-side metric comparison

### Batch Processing

- Process multiple images sequentially
- Real-time progress tracking
- Automatic error handling and recovery
- Summary statistics across entire batch
- Export batch results to comprehensive CSV

## 🔬 Analysis Methods

### Colocalization Metrics

#### 1. Intensity Correlation Quotient (ICQ)
**Range:** -0.5 to +0.5  
**Interpretation:**
- **ICQ > 0.1**: Positive colocalization (proteins co-vary)
- **ICQ ≈ 0**: Random distribution
- **ICQ < -0.1**: Segregation (proteins anti-correlate)

**Formula:**  
`ICQ = (Covariance(I1, I2) - mean(Covariance)) / std(Covariance)`

#### 2. Manders Coefficients (M1, M2)
**Range:** 0 to 1  
**Interpretation:**
- **M1**: Fraction of GFP intensity that colocalizes with mCherry
- **M2**: Fraction of mCherry intensity that colocalizes with GFP
- Values > 0.6 indicate strong colocalization

**Formula:**  
`M1 = Σ(GFP_coloc) / Σ(GFP_total)`  
`M2 = Σ(mCherry_coloc) / Σ(mCherry_total)`

#### 3. Conditional Colocalization Score (CCS)
**Range:** 0 to 1  
**Interpretation:**
- Measures colocalization conditioned on presence of reference structure
- Accounts for expression level differences
- **CCS > 0.5**: Strong conditional colocalization

#### 4. Jaccard Index
**Range:** 0 to 1  
**Interpretation:**
- Measures physical overlap of detected structures
- **0.7+**: Strong overlap
- **0.4-0.7**: Moderate overlap
- **<0.4**: Weak overlap

**Formula:**  
`Jaccard = |A ∩ B| / |A ∪ B|`

#### 5. Enrichment Ratio
**Range:** 0 to ∞  
**Interpretation:**
- Fold-enrichment of signal in structures vs whole cell
- **>1.5x**: Significant enrichment
- **>3x**: Strong active recruitment
- **<1x**: Depletion from structures

**Formula:**  
`Enrichment = (Mean_in_structures / Mean_whole_cell)`

#### 6. Recruitment ICQ
**Range:** -0.5 to +0.5  
**Interpretation:**
- Directional correlation using whole-cell means
- Reveals recruitment direction and strength
- More sensitive than traditional ICQ for recruitment

### Detection Algorithms

#### Granule Detection (LoG-based)
1. Background subtraction (rolling ball)
2. Laplacian of Gaussian filtering
3. Local maxima detection
4. Watershed segmentation
5. Size filtering (min/max thresholds)

#### Colocalization Detection
1. Otsu thresholding per channel
2. Pixel-wise AND operation
3. Connected component analysis
4. Overlap quantification

## 📖 Usage Guide

### 1. Setup Tab - Image Selection

**Select Folder:**
- Click "Browse" button
- Navigate to folder containing .tif or .tiff files
- Application will automatically detect all compatible images
- Supported formats: `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`

**Image Requirements:**
- Two-channel images (GFP and mCherry)
- Minimum recommended size: 512×512 pixels
- 8-bit or 16-bit grayscale per channel
- Files should be properly named for identification

### 2. Parameters Tab - Fine-Tuning Detection

#### Preprocessing Parameters

**Background Radius** (default: 50)
- Size of rolling ball for background subtraction
- Increase (70-100) for images with uneven illumination
- Decrease (30-40) for images with fine structures

**Apply Deconvolution** (default: ON)
- Enhances image sharpness
- Recommended: Keep ON for most applications

#### Granule Detection Parameters

**Min Granule Size** (default: 3 pixels)
- Minimum area for detected structures
- Increase (5-10) to filter out noise
- Decrease (1-2) to detect smaller structures

**Max Granule Size** (default: 30 pixels)
- Maximum area for detected structures
- Increase for larger condensates/organelles
- Decrease for punctate structures

**LoG Threshold** (default: 0.01)
- Sensitivity of Laplacian of Gaussian detector
- **Lower values** (0.001-0.005): More sensitive, more detections
- **Higher values** (0.02-0.05): Less sensitive, fewer false positives

**mCherry Threshold Factor** (default: 1.5)
- Multiplier for mCherry detection sensitivity
- Adjust if mCherry signal is weaker/stronger than GFP

### 3. Single Image Tab - Preview Mode

**Purpose:** Test parameters on single image before batch processing

**Workflow:**
1. Click "Load Single Image" and select test image
2. Adjust parameters using live sliders
3. View real-time detection preview
4. Click "Analyze Image" for full analysis
5. Switch between "Preview" and "Analysis Results" modes

**Use Cases:**
- Parameter optimization
- Quality control
- Exploring different analysis modes
- Educational demonstrations

### 4. Batch Processing

**Steps:**
1. Ensure folder is selected in Setup tab
2. Choose analysis mode (GFP or mCherry)
3. Optionally adjust parameters
4. Click "Process Batch"
5. Monitor progress bar
6. Wait for "Processing complete" message

**Progress Indicators:**
- Progress bar shows completion percentage
- Status bar displays current image being processed
- Log window shows detailed processing information

**Time Estimates:**
- ~5-15 seconds per image (depends on size and complexity)
- Batch of 10 images: ~1-3 minutes
- Batch of 50 images: ~5-15 minutes

### 5. Results Tab - Visualization

**Image Selection:**
- Use dropdown to select specific image
- All processed images available

**Display Options:**

**Analysis Views:**
- 🔬 All Metrics - Comprehensive overview with all metrics
- 📈 Batch Overview - Summary across all images

**Colocalization Types:**
- 🟡 Intensity-Based (Otsu) - Threshold-based colocalization
- 🌐 Whole-Cell ICQ - Global correlation analysis
- 🔬 Granule-Level ICQ - Structure-specific correlation

**Granule Analysis:**
- 📊 Physical Overlap - Venn diagrams and overlap metrics
- 📈 Enrichment Analysis - Fold-enrichment visualization
- 🎯 Recruitment ICQ - Bidirectional recruitment
- 📋 Method Comparison - All methods side-by-side

**Images:**
- 🎨 All Channels - GFP, mCherry, RGB overlay

### 6. Batch Results Tab

**Summary Statistics:**
- Mean ± SD for all metrics
- Sample size information
- Quality assessment

**Results Table:**
- Complete metrics for each image
- Sortable columns
- All colocalization parameters displayed

**Export Options:**
- CSV export with all metrics
- Compatible with Excel, R, Python
- Includes image names and all calculated values

### 7. Exporting Results

**CSV Export:**
```
Columns include:
- Image name
- CCS Mean/Std
- Translocation Efficiency
- ICQ scores
- Recruitment metrics (both directions)
- Enrichment ratios (both directions)
- Jaccard Index
- Manders M1/M2
```

**HTML Reports:**
- Formatted summary report
- Parameter documentation
- Visual summaries (if implemented)

## 📊 Output Metrics

### Standard Metrics per Image

| Metric | Description | Range | Interpretation |
|--------|-------------|-------|----------------|
| **CCS Mean** | Conditional Colocalization Score | 0-1 | >0.5 = strong |
| **Translocation %** | Fraction recruited to structures | 0-100% | >50% = significant |
| **ICQ Mean** | Intensity Correlation Quotient | -0.5 to +0.5 | >0.1 = positive |
| **Recruit→GFP** | mCherry recruitment to GFP | -0.5 to +0.5 | >0.1 = recruitment |
| **Recruit→mCherry** | GFP recruitment to mCherry | -0.5 to +0.5 | >0.1 = recruitment |
| **Enrichment mCh→GFP** | mCherry fold-enrichment in GFP | 0-∞ | >1.5x = enriched |
| **Enrichment GFP→mCh** | GFP fold-enrichment in mCherry | 0-∞ | >1.5x = enriched |
| **Jaccard Index** | Physical structure overlap | 0-1 | >0.6 = strong |
| **Manders M1** | GFP colocalization fraction | 0-1 | >0.6 = strong |
| **Manders M2** | mCherry colocalization fraction | 0-1 | >0.6 = strong |

### Interpretation Guidelines

#### Strong Colocalization Pattern
- ICQ > 0.2
- Manders M1, M2 > 0.6
- Jaccard > 0.6
- Enrichment > 2x

#### Asymmetric Recruitment
- |Recruit→A - Recruit→B| > 0.2
- Different enrichment ratios
- Dominant direction clearly visible

#### No Colocalization
- ICQ < 0.05
- Manders M1, M2 < 0.3
- Jaccard < 0.2
- Enrichment < 1.2x

## 💻 Requirements

### System Requirements

**Minimum:**
- CPU: Dual-core processor
- RAM: 4 GB
- Storage: 100 MB for application + space for images
- Display: 1280×720 resolution

**Recommended:**
- CPU: Quad-core processor or better
- RAM: 8 GB or more
- Storage: 1 GB free space
- Display: 1920×1080 or higher

### Python Version
- Python 3.8, 3.9, 3.10, or 3.11
- Not tested on Python 3.12+ (may work)

### Operating Systems
- ✅ Windows 10/11
- ✅ macOS 10.14+
- ✅ Linux (Ubuntu 20.04+, other distributions)

## 🐛 Troubleshooting

### Common Issues

#### "Module not found" error
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

#### Images not loading
- Check file format (.tif, .tiff supported)
- Ensure images are two-channel
- Verify file permissions

#### "No granules detected"
- Lower LoG Threshold (try 0.001-0.005)
- Adjust Background Radius
- Check if image actually contains structures
- Try Single Image mode to preview detection

#### Application crashes during processing
- Check available RAM (close other applications)
- Process smaller batches
- Reduce image size if very large (>4000×4000)

#### Slow performance
- Reduce batch size
- Process images sequentially rather than loading all
- Use SSD storage for image files
- Close other applications

### Getting Help

1. Check [Issues](https://github.com/YOUR_USERNAME/fluorescence-colocalization-analyzer/issues) page
2. Search closed issues for solutions
3. Create new issue with:
   - Detailed problem description
   - Screenshots
   - Error messages
   - System information

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
2. **Create feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit changes** (`git commit -m 'Add AmazingFeature'`)
4. **Push to branch** (`git push origin feature/AmazingFeature`)
5. **Open Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/fluorescence-colocalization-analyzer.git
cd fluorescence-colocalization-analyzer

# Create development environment
python -m venv dev-env
source dev-env/bin/activate  # or dev-env\Scripts\activate on Windows

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt  # if available
```

### Code Style

- Follow PEP 8 guidelines
- Use descriptive variable names
- Add docstrings to functions
- Comment complex algorithms
- Keep functions focused and modular

### Areas for Contribution

- 🐛 Bug fixes
- ✨ New analysis metrics
- 📊 Additional visualization types
- 🎨 UI/UX improvements
- 📝 Documentation enhancements
- 🧪 Unit tests
- 🌍 Translations

## 📄 License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 📚 Citation

If you use this software in your research, please cite:

```bibtex
@software{fluorescence_colocalization_analyzer,
  author = {[Your Name]},
  title = {Fluorescence Microscopy Colocalization Analyzer},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/fluorescence-colocalization-analyzer},
  version = {1.0.0}
}
```

## 🙏 Acknowledgments

- Developed for quantitative analysis of protein colocalization in fluorescence microscopy
- Built with Python, NumPy, SciPy, and Matplotlib
- Inspired by ImageJ colocalization plugins and modern granule biology research

## 📞 Contact

**Project Link:** [https://github.com/YOUR_USERNAME/fluorescence-colocalization-analyzer](https://github.com/YOUR_USERNAME/fluorescence-colocalization-analyzer)

**Issues:** [https://github.com/YOUR_USERNAME/fluorescence-colocalization-analyzer/issues](https://github.com/YOUR_USERNAME/fluorescence-colocalization-analyzer/issues)

---

**⭐ If you find this tool useful, please consider starring the repository!**
