# Traffic Flow Monitor

Simple vehicle detection and counting project using YOLOv8 and OpenCV.

## Requirements

- Python 3.10+
- ultralytics
- opencv-python
- Model weights (for example: `yolov8s.pt`)

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python src/main.py
```

## Limitations

- Accuracy drops in low light, rain, glare, and heavy occlusion.
- Static ROI mask is camera-specific and must be redrawn for new views.
- ID switches in tracking can cause missed counts or double counts.
- Current setup is best for mostly one-direction flow near a single counting line.

## Improvements

- Fine-tune YOLOv8 on local traffic data.
- Add directional and multi-lane counting logic.
- Add evaluation against manually labeled ground truth.
- Optimize for real-time edge deployment.

## License

YOLOv8 is licensed under AGPL-3.0 (Ultralytics). Ensure compliance when deploying or distributing.

## Resources

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://opencv.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)
