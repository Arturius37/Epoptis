# Epoptis — Live Vision

A browser-based camera experience for real-time object recognition and visual-change analysis, with a quiet, layered light-grey interface and a focus on local processing.

## Development status

**An honest note about recognition:** We’re sorry that Epoptis — Live Vision still misses some everyday objects and may occasionally identify an item incorrectly. Recognition is not as broad or reliable as we want it to be yet. We’re actively working to improve object coverage, accuracy, and consistency across different objects, angles, and lighting conditions. Thank you for your patience and for trying the project while these improvements are in progress.

Please treat detections as experimental rather than guaranteed results.

## Run locally

Camera access requires a secure browser context (`localhost` or HTTPS). With Node.js installed:

```sh
npx serve .
```
## Run locally second way

Copy this and paste in any browser. And please wait there for downloading models.

```sh
https://epoptis.netlify.app/
```
Open the local URL in a modern browser and allow camera access. An internet connection is required to download the JavaScript runtimes and model weights on first use. The models are cached by the browser where supported.

## Features

- **Live object detection:** COCO-SSD MobileNetV2 draws detection boxes and confidence values for its trained classes.
- **Extended everyday-item detection:** An open-vocabulary OWL-ViT detector searches a broader built-in vocabulary of everyday objects, plus custom labels saved in Settings. User-defined labels are prioritized so they remain in the query set. The slower extended pass runs alongside the fast COCO-SSD detector rather than pausing its updates.
- **Custom search labels:** Use Settings → “Extra objects to recognize” to add a comma-separated set of extra category names.
- **Readable front-camera overlays:** The front-camera image may be mirrored for a familiar selfie preview, but detection labels stay upright and readable.
- **Teach an item:** Save a local name and visual embedding for a detected item; future crops can be compared with the saved example in this browser's local storage.
- **Motion analysis:** Frame differences estimate how much of the image is changing and visualize changed regions. Camera shake, shadows, and lighting changes can affect results.
- **Scene reference:** Save a view as a baseline and compare later frames against it. This measures pixel-level visual change, not the meaning or cause of a change.
- **Session controls:** Switch camera where supported, pause/resume analysis, stop the camera, capture a snapshot, toggle motion/object detection, and adjust sensitivity.
- **Local image processing:** Camera frames are processed in the browser and are not uploaded by the app. Only the JavaScript runtime and model assets are downloaded from their hosting services.
- **Responsive interface:** Supports desktop and mobile layouts, keyboard focus styling, and reduced-motion preferences.

## Recognition limitations

No general model recognizes every object perfectly. Accuracy depends on the selected labels, lighting, focus, object size, angle, and how much of the object is visible. Bring small objects such as pens and glasses close to the camera and make sure they are well lit. The extended detector downloads an additional model of approximately 127–155 MB on first use, depending on the available device backend; if it fails to load or runs too slowly, turn off **Extended** and use the faster standard detector. Standard detection still works without that additional model.

Camera access requires HTTPS or `localhost`. CDN-hosted code and model weights require internet access unless you later self-host them.
