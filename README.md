# Eyes Defy Anemia - AI Medical Analysis Platform

AI-powered anemia detection system using conjunctival pallor analysis from eye images. Includes both machine learning training and professional web interface.

## Features

### Web Application
- Professional medical interface with drag & drop image upload
- Real-time patient form validation
- Interactive anemia analysis simulation
- Animated results display with hemoglobin levels
- Medical report generation and sharing

### Machine Learning
- PyTorch CNN for anemia classification
- Cross-validation training with data augmentation
- **Actual Performance**: 74.6% accuracy, 1.26 MAE
- Trained on 218 medical samples from India/Italy datasets

## Quick Start

### Web App
```bash
# Simply open index.html in your browser
open index.html
```

### Train Model
```bash
# Install dependencies
pip install torch torchvision opencv-python pandas numpy scikit-learn matplotlib seaborn

# Run training
python improved_final_pipeline.py
```

## Files Included

```
├── index.html                    # Professional web interface
├── styles.css                   # Medical-grade styling
├── script.js                    # Interactive functionality
├── improved_final_pipeline.py   # Final ML training pipeline
└── README.md                    # This file
```

## Model Performance

- **Accuracy**: 74.6%
- **MAE (Mean Absolute Error)**: 1.26
- **Training Samples**: 218 medical images
- **Classes**: Anemia severity levels (Mild/Moderate/Severe)

## Medical Disclaimer

**For educational and research purposes only.** This tool is not intended for actual medical diagnosis. Always consult healthcare professionals for medical advice.

## Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **ML**: PyTorch, OpenCV, scikit-learn
- **Data**: Medical eye images with conjunctival pallor analysis
