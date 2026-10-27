(() => {
  const $ = (id) => document.getElementById(id);
  const video = $('video'), canvas = $('overlay'), ctx = canvas.getContext('2d');
  const motionCanvas = $('motionCanvas'), motionCtx = motionCanvas.getContext('2d', { willReadFrequently: true });
  let stream = null, model = null, detailModel = null, detailModelPromise = null;
  let extendedModel = null, extendedModelPromise = null, extendedBusy = false;
  let running = false, paused = false, facing = 'user';
  let lastFrame = null, lastDetections = [], standardDetections = [], extendedDetections = [];
  let lastInfer = 0, lastExtendedInfer = 0, lastMotion = 0, lastTimestamp = 0, fps = 0;
  let lastDetail = 0, detailBusy = false, detailCursor = 0;
  let referenceFrame = null, motionThreshold = 28, confidenceThreshold = 0.30, startAttempt = 0;
  const colors = ['#b9d0c9','#b8c7d9','#d6cbb3','#c2bfdc','#dfc0ca'];
  // Broad, practical vocabulary for the open-vocabulary detector. Keep labels
  // as short category names because OWL-ViT accepts candidate labels directly.
  const defaultExtendedLabels = [
    'pen', 'ballpoint pen', 'writing pen', 'pencil', 'marker', 'highlighter', 'ruler',
    'eraser', 'stapler', 'scissors', 'book', 'notebook', 'paper', 'folder', 'calculator',
    'glasses', 'eyeglasses', 'spectacles', 'reading glasses', 'sunglasses', 'bottle',
    'plastic bottle', 'water bottle', 'thermos', 'cup', 'mug', 'jar', 'can', 'food container',
    'lunchbox', 'bowl', 'plate', 'spoon', 'fork', 'remote control', 'computer mouse',
    'keyboard', 'monitor', 'laptop', 'tablet', 'smartphone', 'mobile phone', 'cell phone',
    'camera', 'headphones', 'earbuds', 'charger', 'phone charger', 'charging cable',
    'USB cable', 'USB flash drive', 'power bank', 'power adapter', 'game controller', 'watch',
    'keys', 'wallet', 'backpack', 'handbag', 'suitcase', 'shoe', 'slipper', 'hat', 'umbrella',
    'flashlight', 'battery', 'screwdriver', 'hammer', 'toothbrush', 'comb'
  ];
  const extendedLabelAliases = {
    'ballpoint pen': 'pen', 'writing pen': 'pen',
    'eyeglasses': 'glasses', 'spectacles': 'glasses', 'reading glasses': 'glasses',
    'plastic bottle': 'bottle', 'water bottle': 'bottle',
    'mobile phone': 'smartphone', 'cell phone': 'smartphone',
    'phone charger': 'charger', 'usb cable': 'charging cable'
  };
  function normalizeExtendedLabel(label) {
    const cleaned = String(label || '')
      .replace(/^\s*a photo of\s+/i, '')
      .replace(/^\s*(?:an?|the)\s+/i, '')
      .trim().toLowerCase();
    return Object.prototype.hasOwnProperty.call(extendedLabelAliases, cleaned)
      ? extendedLabelAliases[cleaned]
      : cleaned;
  }
  const extendedCanvas = document.createElement('canvas');
  const extendedCtx = extendedCanvas.getContext('2d', { willReadFrequently: true });
  let userExtraLabels = loadExtraObjectLabels();
  let detailResult = null;
  let taughtItems = loadTaughtItems(), teachingTarget = null;

  function loadTaughtItems() { try { return JSON.parse(localStorage.getItem('fieldnote-taught-items') || '[]'); } catch { return []; } }
  function loadExtraObjectLabels() { try { return String(localStorage.getItem('fieldnote-extra-object-labels') || '').split(',').map(s => s.trim()).filter(Boolean).slice(0, 12); } catch { return []; } }
  function getExtendedLabels() {
    // Put user-defined labels first so the built-in vocabulary never crowds
    // out an item the person explicitly asked the camera to look for.
    const labels = [...userExtraLabels, ...defaultExtendedLabels]
      .map(label => label.trim().toLowerCase())
      .filter(Boolean);
    return [...new Set(labels)].slice(0, 82);
  }
  function cropBox(target) {
    const [x,y,w,h] = target.bbox;
    const crop = document.createElement('canvas'); crop.width = 224; crop.height = 224;
    const sx = Math.max(0, x), sy = Math.max(0, y);
    crop.getContext('2d').drawImage(video, sx, sy, Math.max(1, Math.min(w, video.videoWidth - sx)), Math.max(1, Math.min(h, video.videoHeight - sy)), 0, 0, 224, 224);
    return crop;
  }
  function normalizedEmbedding(image) {
    const tensor = detailModel.infer(image, true), vector = Array.from(tensor.dataSync()); tensor.dispose();
    const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)) || 1;
    return vector.map(value => value / norm);
  }
  function findTaughtMatch(vector) {
    let best = null;
    for (const sample of taughtItems) {
      if (!sample.vector || sample.vector.length !== vector.length) continue;
      let similarity = 0;
      for (let i = 0; i < vector.length; i++) similarity += vector[i] * sample.vector[i];
      if (!best || similarity > best.score) best = { name: sample.name, score: similarity };
    }
    return best;
  }
  function status(text, mode = text) { $('statusText').textContent = text.toUpperCase(); $('modeLabel').textContent = mode.toUpperCase(); }
  function toast(message) { const el = $('toast'); el.textContent = message; el.classList.add('show'); setTimeout(() => el.classList.remove('show'), 2600); }
  function sizeCanvases() {
    const rect = $('cameraFrame').getBoundingClientRect();
    const dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(rect.width * dpr); canvas.height = Math.round(rect.height * dpr);
    canvas.style.width = `${rect.width}px`; canvas.style.height = `${rect.height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  function drawDetections(detections) {
    const rect = $('cameraFrame').getBoundingClientRect();
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return;
    // Map source coordinates to the object-fit:cover preview. Only the box
    // position is mirrored for the front camera; its text is drawn upright.
    const scale = Math.max(rect.width / vw, rect.height / vh);
    const ox = (rect.width - vw * scale) / 2, oy = (rect.height - vh * scale) / 2;
    ctx.save();
    detections.forEach((item, idx) => {
      const [x, y, w, h] = item.bbox;
      const sourceX = ox + x * scale, by = oy + y * scale, bw = w * scale, bh = h * scale;
      const bx = facing === 'user' ? rect.width - (sourceX + bw) : sourceX;
      const color = colors[idx % colors.length];
      ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.strokeRect(bx, by, bw, bh);
      const detail = detailResult && detailResult.class === item.class && Math.hypot(
        (detailResult.bbox[0] + detailResult.bbox[2] / 2) - (item.bbox[0] + item.bbox[2] / 2),
        (detailResult.bbox[1] + detailResult.bbox[3] / 2) - (item.bbox[1] + item.bbox[3] / 2)
      ) < Math.max(90, Math.hypot(item.bbox[2], item.bbox[3]) * .6) ? detailResult : null;
      const baseLabel = `${item.class.toUpperCase()} ${Math.round(item.score * 100)}%`;
      // Generic classifier guesses are hidden; only user-taught matches can
      // add a second label, avoiding plausible-looking but incorrect names.
      const detailLabel = detail && detail.name.toLowerCase() !== item.class.toLowerCase()
        ? detail.kind === 'learned' && detail.score >= .82
          ? ` · MY ITEM: ${detail.name.toUpperCase()} (${Math.round(detail.score * 100)} MATCH)`
          : detail.kind === 'possible' ? ` · POSSIBLE: ${detail.name.toUpperCase()} (${Math.round(detail.score * 100)} MATCH)`
          : ''
        : '';
      const label = baseLabel + detailLabel;
      ctx.font = '10px "DM Mono", monospace';
      const tw = Math.min(rect.width - 12, ctx.measureText(label).width + 14);
      const labelX = Math.max(0, Math.min(bx, rect.width - tw - 4));
      const labelY = Math.max(0, by - 21);
      ctx.fillStyle = color; ctx.fillRect(labelX, labelY, tw, 20);
      ctx.fillStyle = '#101814'; ctx.fillText(label, labelX + 7, labelY + 13, tw - 12);
      const cx = bx + bw / 2, cy = by + bh / 2;
      ctx.beginPath(); ctx.arc(cx, cy, 2.3, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
    });
    ctx.restore();
  }

  function intersectionOverUnion(a, b) {
    const left = Math.max(a[0], b[0]), top = Math.max(a[1], b[1]);
    const right = Math.min(a[0] + a[2], b[0] + b[2]);
    const bottom = Math.min(a[1] + a[3], b[1] + b[3]);
    const area = Math.max(0, right - left) * Math.max(0, bottom - top);
    const union = a[2] * a[3] + b[2] * b[3] - area;
    return union > 0 ? area / union : 0;
  }

  function refreshCombinedDetections() {
    if (!$('objectToggle').checked) {
      lastDetections = [];
      $('objectCount').textContent = 'OFF';
      $('objectHint').textContent = 'Object detection paused';
      $('teachButton').disabled = true;
      return;
    }
    const candidates = [...standardDetections, ...extendedDetections]
      .filter(item => item && item.bbox && item.bbox.every(Number.isFinite) && Number.isFinite(item.score))
      .sort((a, b) => b.score - a.score);
    const merged = [];
    for (const candidate of candidates) {
      const duplicate = merged.some(existing => {
        const overlap = intersectionOverUnion(existing.bbox, candidate.bbox);
        return overlap > .52 && (existing.class.toLowerCase() === candidate.class.toLowerCase() || overlap > .82);
      });
      if (!duplicate) merged.push(candidate);
    }
    lastDetections = merged.slice(0, 20);
    $('objectCount').textContent = String(lastDetections.length);
    $('objectHint').textContent = lastDetections.length
      ? [...new Set(lastDetections.map(d => d.class))].slice(0, 3).join(' · ')
      : (extendedModelPromise && $('extendedToggle').checked ? 'Loading wider item detector…' : 'No objects above threshold — try moving closer');
    $('teachButton').disabled = !lastDetections.length || !detailModel;
  }

  async function ensureExtendedModel() {
    if (extendedModel) return true;
    if (extendedModelPromise) return extendedModelPromise;
    extendedModelPromise = (async () => {
      toast('Loading extended recognition. First use downloads an additional model (about 155 MB).');
      status('Loading extended model', 'Wider object detection');
      try {
        const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1');
        if (env) env.useBrowserCache = true;
        // Prefer a quantized WebGPU build when the browser can provide an
        // adapter; gracefully fall back to the wider CPU/WASM-compatible build.
        if (navigator.gpu?.requestAdapter) {
          try {
            const adapter = await navigator.gpu.requestAdapter();
            if (adapter) extendedModel = await pipeline('zero-shot-object-detection', 'Xenova/owlvit-base-patch32', { device: 'webgpu', dtype: 'q4f16' });
          } catch (gpuError) { console.warn('WebGPU unavailable for extended recognition; trying CPU mode:', gpuError); }
        }
        if (!extendedModel) extendedModel = await pipeline('zero-shot-object-detection', 'Xenova/owlvit-base-patch32', { device: 'wasm', dtype: 'q8' });
        if (running) status(paused ? 'Paused' : 'Analyzing live', paused ? 'Analysis paused' : 'Live analysis');
        toast('Extended recognition is ready. Small items work best close to the camera.');
        return true;
      } catch (error) {
        console.error('Extended object detector failed to load:', error);
        $('extendedToggle').checked = false;
        extendedDetections = [];
        refreshCombinedDetections();
        if (running) status(paused ? 'Paused' : 'Analyzing live', paused ? 'Analysis paused' : 'Live analysis');
        toast('Extended detector could not load. Standard recognition remains available.');
        return false;
      } finally {
        extendedModelPromise = null;
      }
    })();
    return extendedModelPromise;
  }

  async function runExtendedRecognition(now) {
    if (!running || paused || !model || !extendedModel || extendedBusy || detailBusy ||
        !$('objectToggle').checked || !$('extendedToggle').checked || video.readyState < 2 ||
        now - lastExtendedInfer < 1800) return;
    lastExtendedInfer = now;
    extendedBusy = true;
    try {
      // Analyze a downscaled canvas locally to keep the second detector practical.
      const sourceWidth = video.videoWidth, sourceHeight = video.videoHeight;
      if (!sourceWidth || !sourceHeight) return;
      // Preserve more detail for small everyday items while keeping the longest
      // side bounded for phones and laptops. Avoid downscaling twice before the
      // detector's own preprocessing step.
      const inputScale = Math.min(768 / sourceWidth, 768 / sourceHeight, 1);
      const inputWidth = Math.max(1, Math.round(sourceWidth * inputScale));
      const inputHeight = Math.max(1, Math.round(sourceHeight * inputScale));
      extendedCanvas.width = inputWidth;
      extendedCanvas.height = inputHeight;
      extendedCtx.drawImage(video, 0, 0, inputWidth, inputHeight);
      // OWL-ViT commonly returns lower raw scores than COCO-SSD. Use a modestly
      // lower default threshold to recover small/fine-grained items; the existing
      // confidence slider still raises the threshold for users preferring precision.
      const threshold = Math.max(.045, Math.min(.18, confidenceThreshold * .22));
      const results = await extendedModel(extendedCanvas, getExtendedLabels(), { threshold, top_k: 18 });
      if (!running || !$('objectToggle').checked || !$('extendedToggle').checked) return;
      const sx = sourceWidth / inputWidth, sy = sourceHeight / inputHeight;
      extendedDetections = (Array.isArray(results) ? results : []).map(result => {
        const box = result.box || {};
        const x1 = Math.max(0, Math.min(sourceWidth, Number(box.xmin) * sx));
        const y1 = Math.max(0, Math.min(sourceHeight, Number(box.ymin) * sy));
        const x2 = Math.max(x1, Math.min(sourceWidth, Number(box.xmax) * sx));
        const y2 = Math.max(y1, Math.min(sourceHeight, Number(box.ymax) * sy));
        return {
          class: normalizeExtendedLabel(result.label),
          score: Number(result.score) || 0,
          bbox: [x1, y1, x2 - x1, y2 - y1],
          source: 'extended'
        };
      }).filter(item => item.class && item.score >= threshold && item.bbox[2] >= 6 && item.bbox[3] >= 6);
      refreshCombinedDetections();
    } catch (error) {
      console.warn('Extended detection pass skipped:', error);
    } finally {
      extendedBusy = false;
      lastExtendedInfer = performance.now();
    }
  }

  function analyzeMotion(now) {
    if (now - lastMotion < 115 || video.readyState < 2) return;
    lastMotion = now;
    const w = 160, h = Math.max(90, Math.round(160 * video.videoHeight / video.videoWidth));
    motionCanvas.width = w; motionCanvas.height = h;
    motionCtx.save();
    if (facing === 'user') { motionCtx.translate(w, 0); motionCtx.scale(-1, 1); }
    motionCtx.drawImage(video, 0, 0, w, h); motionCtx.restore();
    const current = motionCtx.getImageData(0, 0, w, h).data;
    const gray = new Uint8Array(w * h);
    for (let i = 0, p = 0; i < current.length; i += 4, p++) gray[p] = (current[i] * 3 + current[i + 1] * 6 + current[i + 2]) / 10;

    if (lastFrame) {
      const heat = new Uint8Array(w * h);
      let moving = 0, sx = 0, sy = 0, sampleCount = 0, changedFromReference = 0;
      const hasReference = referenceFrame && referenceFrame.length === gray.length;
      for (let y = 1; y < h - 1; y += 2) {
        for (let x = 1; x < w - 1; x += 2) {
          const p = y * w + x;
          const delta = Math.abs(gray[p] - lastFrame[p]);
          sampleCount++;
          if (delta > motionThreshold) { heat[p] = 1; moving++; sx += x; sy += y; }
          if (hasReference && Math.abs(gray[p] - referenceFrame[p]) > motionThreshold) changedFromReference++;
        }
      }
      const motionPercent = sampleCount ? moving / sampleCount * 100 : 0;
      $('motionValue').textContent = `${motionPercent.toFixed(1)}%`;
      $('motionHint').textContent = moving > 12 ? `${moving} changing sample points` : 'Scene mostly still';

      if (hasReference) {
        const scenePercent = sampleCount ? changedFromReference / sampleCount * 100 : 0;
        $('sceneValue').textContent = `${scenePercent.toFixed(1)}%`;
        $('sceneHint').textContent = scenePercent >= 24 ? 'Large visual change' : scenePercent >= 9 ? 'Noticeable difference' : 'Close to saved reference';
      } else {
        $('sceneValue').textContent = '—';
        $('sceneHint').textContent = 'Set a reference frame';
      }

      if ($('motionToggle').checked) {
        const rect = $('cameraFrame').getBoundingClientRect();
        // Use the same object-fit: cover mapping as the video, so the overlay
        // stays aligned even when the camera and preview aspect ratios differ.
        const scale = Math.max(rect.width / w, rect.height / h);
        const ox = (rect.width - w * scale) / 2, oy = (rect.height - h * scale) / 2;
        ctx.save(); ctx.fillStyle = '#759b97';
        for (let y = 2; y < h - 1; y += 3) {
          for (let x = 2; x < w - 1; x += 3) {
            if (heat[y * w + x]) ctx.fillRect(ox + x * scale, oy + y * scale, Math.max(1, scale * 1.25), Math.max(1, scale * 1.25));
          }
        }
        if (moving > 35) {
          const cx = ox + sx / moving * scale, cy = oy + sy / moving * scale;
          const dx = cx - rectCenterX(), dy = cy - rectCenterY();
          const len = Math.hypot(dx, dy) || 1;
          ctx.strokeStyle = '#759b97'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + dx / len * 34, cy + dy / len * 34); ctx.stroke();
          ctx.beginPath(); ctx.arc(cx, cy, 8, 0, Math.PI * 2); ctx.stroke();
        }
        ctx.restore();
      }
    }
    lastFrame = gray;
  }
  function setSceneReference() {
    if (!lastFrame) { toast('Wait for the live camera frame, then set a reference.'); return; }
    referenceFrame = new Uint8Array(lastFrame);
    $('sceneValue').textContent = '0.0%';
    $('sceneHint').textContent = 'Reference saved';
    $('baselineButton').innerHTML = '<span>⌖</span> UPDATE REF';
    toast('Scene reference saved on this device.');
  }
  function rectCenterX() { return $('cameraFrame').getBoundingClientRect().width / 2; }
  function rectCenterY() { return $('cameraFrame').getBoundingClientRect().height / 2; }

  async function loop(now) {
    if (!running) return;
    if (paused || video.readyState < 2) { requestAnimationFrame(loop); return; }
    sizeCanvases(); ctx.clearRect(0, 0, canvas.width, canvas.height);
    analyzeMotion(now);
    if ($('objectToggle').checked && $('extendedToggle').checked && !extendedModel && !extendedModelPromise) void ensureExtendedModel();
    void runExtendedRecognition(now);
    // Keep the fast detector live while the slower open-vocabulary pass is
    // running; extended inference must not freeze standard object updates.
    if ($('objectToggle').checked && model && now - lastInfer > 350) {
      lastInfer = now;
      try {
        const detections = await model.detect(video, 20, confidenceThreshold);
        standardDetections = detections.map(item => ({ ...item, bbox: [...item.bbox], source: 'standard' }));
      } catch (e) { console.warn('Detection frame skipped', e); }
      refreshCombinedDetections();
      if (detailModel && lastDetections.length && !detailBusy && !extendedBusy && now - lastDetail > 1500) {
        lastDetail = now;
        const target = lastDetections[detailCursor % lastDetections.length];
        detailCursor++;
        detailBusy = true;
        try {
          const crop = cropBox(target);
          if (taughtItems.length) {
            const match = findTaughtMatch(normalizedEmbedding(crop));
            if (match && match.score >= .76) detailResult = { name: match.name, score: match.score, kind: match.score >= .82 ? 'learned' : 'possible', class: target.class, bbox: [...target.bbox] };
            else {
              const result = await detailModel.classify(crop, 3);
              if (result?.length) detailResult = { name: result[0].className.split(',')[0], score: result[0].probability, kind: 'model', class: target.class, bbox: [...target.bbox] };
            }
          } else {
            const result = await detailModel.classify(crop, 3);
            if (result?.length) detailResult = { name: result[0].className.split(',')[0], score: result[0].probability, kind: 'model', class: target.class, bbox: [...target.bbox] };
          }
        } catch (e) { console.warn('Detailed classification skipped', e); }
        finally { detailBusy = false; }
      }
    } else if (!$('objectToggle').checked) {
      standardDetections = []; extendedDetections = []; refreshCombinedDetections();
    }
    drawDetections(lastDetections);
    if (lastTimestamp) { const instant = 1000 / Math.max(1, now - lastTimestamp); fps = fps ? fps * .82 + instant * .18 : instant; $('fpsValue').textContent = `${Math.round(fps)}`; }
    lastTimestamp = now;
    if (running) requestAnimationFrame(loop);
  }

  async function startCamera() {
    const thisAttempt = ++startAttempt;
    if (!navigator.mediaDevices?.getUserMedia) { toast('Camera needs HTTPS or localhost in a supported browser.'); return; }
    running = false;
    lastFrame = null; referenceFrame = null; detailResult = null; lastDetections = []; standardDetections = []; extendedDetections = [];
    $('sceneValue').textContent = '—'; $('sceneHint').textContent = 'Set a reference frame';
    $('baselineButton').innerHTML = '<span>⌖</span> REFERENCE';
    $('switchButton').disabled = $('pauseButton').disabled = $('captureButton').disabled = $('baselineButton').disabled = $('stopButton').disabled = true;
    $('welcome').classList.add('hidden'); $('loadingState').classList.remove('hidden'); $('loadingText').textContent = 'Requesting camera access…';
    $('liveBadge').classList.remove('live'); $('analysisLabel').textContent = 'INITIALIZING';
    status('Connecting', 'Opening camera');
    let requestedStream = null;
    try {
      if (stream) stream.getTracks().forEach(t => t.stop());
      stream = null;
      requestedStream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } } });
      if (thisAttempt !== startAttempt) { requestedStream.getTracks().forEach(t => t.stop()); return; }
      stream = requestedStream;
      video.srcObject = stream; await video.play();
      if (thisAttempt !== startAttempt) { requestedStream.getTracks().forEach(t => t.stop()); return; }
      video.style.display = 'block';
      video.style.transform = facing === 'user' ? 'scaleX(-1)' : 'none';
      $('cameraFacing').textContent = facing === 'user' ? 'FRONT' : 'REAR';
      $('resolution').textContent = `${video.videoWidth} × ${video.videoHeight}`;
      $('cameraId').textContent = facing === 'user' ? '01' : '02';
      $('stopButton').disabled = false;
      $('loadingText').textContent = 'Loading live object recognition…';
      status('Loading model', 'Initializing vision');
      if (!model) model = await cocoSsd.load({ base: 'mobilenet_v2' });
      if (thisAttempt !== startAttempt) return;
      $('loadingState').classList.add('hidden'); running = true; paused = false;
      $('liveBadge').classList.add('live'); $('analysisLabel').textContent = 'ANALYSIS ACTIVE';
      $('switchButton').disabled = $('pauseButton').disabled = $('captureButton').disabled = $('baselineButton').disabled = $('stopButton').disabled = false;
      $('pauseButton').innerHTML = '<span>Ⅱ</span> PAUSE';
      status('Analyzing live', 'Live analysis');
      if (!detailModel && !detailModelPromise) {
        detailModelPromise = mobilenet.load({ version: 2, alpha: 0.5 }).then(loaded => { detailModel = loaded; })
          .catch(e => console.warn('Detail model unavailable; continuing with object detector', e));
      }
      requestAnimationFrame(loop);
      if ($('objectToggle').checked && $('extendedToggle').checked) void ensureExtendedModel();
    } catch (error) {
      if (thisAttempt !== startAttempt) {
        if (requestedStream) requestedStream.getTracks().forEach(t => t.stop());
        return;
      }
      console.error(error); running = false;
      if (requestedStream) requestedStream.getTracks().forEach(t => t.stop());
      stream = null; video.srcObject = null; video.style.display = 'none';
      $('loadingState').classList.add('hidden'); $('welcome').classList.remove('hidden');
      $('liveBadge').classList.remove('live'); $('analysisLabel').textContent = 'ANALYSIS STANDBY';
      $('switchButton').disabled = $('pauseButton').disabled = $('captureButton').disabled = $('baselineButton').disabled = $('stopButton').disabled = true;
      $('teachButton').disabled = true;
      status('Camera unavailable', 'Standby');
      const msg = error.name === 'NotAllowedError' ? 'Allow camera access in your browser, then try again.' : error.name === 'NotFoundError' ? 'No camera was found on this device.' : `Could not start camera: ${error.message}`;
      toast(msg);
    }
  }
  function stopCamera() {
    startAttempt++;
    running = false; if (stream) stream.getTracks().forEach(t => t.stop()); stream = null;
    video.pause(); video.srcObject = null; video.style.display = 'none';
    $('welcome').classList.remove('hidden'); $('loadingState').classList.add('hidden'); $('liveBadge').classList.remove('live'); $('analysisLabel').textContent = 'ANALYSIS STANDBY';
      $('switchButton').disabled = $('pauseButton').disabled = $('captureButton').disabled = $('baselineButton').disabled = $('stopButton').disabled = true;
      $('teachButton').disabled = true;
    $('resolution').textContent = 'NO SIGNAL'; $('objectCount').textContent = '—'; $('motionValue').textContent = '—'; $('sceneValue').textContent = '—'; $('fpsValue').textContent = '—';
    $('objectHint').textContent = 'Awaiting camera'; $('motionHint').textContent = 'Awaiting camera'; $('sceneHint').textContent = 'Set a reference frame';
    status('System ready', 'Standby'); ctx.clearRect(0, 0, canvas.width, canvas.height); lastFrame = null; referenceFrame = null; detailResult = null; lastDetections = []; standardDetections = []; extendedDetections = []; $('baselineButton').innerHTML = '<span>⌖</span> REFERENCE';
  }
  $('startButton').addEventListener('click', startCamera);
  $('switchButton').addEventListener('click', async () => { facing = facing === 'user' ? 'environment' : 'user'; lastFrame = null; referenceFrame = null; await startCamera(); });
  $('baselineButton').addEventListener('click', setSceneReference);
  $('stopButton').addEventListener('click', stopCamera);
  $('pauseButton').addEventListener('click', () => { paused = !paused; $('pauseButton').innerHTML = paused ? '<span>▶</span> RESUME' : '<span>Ⅱ</span> PAUSE'; $('analysisLabel').textContent = paused ? 'ANALYSIS PAUSED' : 'ANALYSIS ACTIVE'; status(paused ? 'Paused' : 'Analyzing live', paused ? 'Analysis paused' : 'Live analysis'); });
  $('captureButton').addEventListener('click', () => {
    if (!video.videoWidth) return;
    const out = document.createElement('canvas'); out.width = video.videoWidth; out.height = video.videoHeight;
    const c = out.getContext('2d'); c.save(); if (facing === 'user') { c.translate(out.width, 0); c.scale(-1, 1); }
    c.drawImage(video, 0, 0); c.restore();
    const a = document.createElement('a'); a.download = `fieldnote-${new Date().toISOString().replace(/[:.]/g, '-')}.jpg`; a.href = out.toDataURL('image/jpeg', .92); a.click(); toast('Snapshot saved to your device.');
  });
  $('motionToggle').addEventListener('change', () => { if (!running) return; });
  $('objectToggle').addEventListener('change', () => {
    if (!$('objectToggle').checked) { standardDetections = []; extendedDetections = []; detailResult = null; }
    refreshCombinedDetections();
    if ($('objectToggle').checked && $('extendedToggle').checked && running) void ensureExtendedModel();
  });
  $('extendedToggle').addEventListener('change', () => {
    if ($('extendedToggle').checked) {
      if (running && $('objectToggle').checked) void ensureExtendedModel();
      else toast('Extended recognition will load when you start the camera.');
    } else {
      extendedDetections = []; refreshCombinedDetections();
      toast('Extended recognition paused. Standard objects remain active.');
    }
  });
  $('teachButton').addEventListener('click', () => {
    teachingTarget = lastDetections[0];
    if (!teachingTarget || !detailModel) return;
    $('teachDescription').textContent = `The current object looks like “${teachingTarget.class}”. Name it once; you can save more views later for better matching.`;
    $('itemName').value = ''; $('teachDialog').showModal(); $('itemName').focus();
  });
  $('cancelTeach').addEventListener('click', () => $('teachDialog').close());
  $('teachForm').addEventListener('submit', (event) => {
    event.preventDefault();
    const name = $('itemName').value.trim();
    if (!name || !teachingTarget || !detailModel) return;
    try {
      if (taughtItems.length >= 80) { toast('Item memory is full. Remove browser site data to start fresh.'); return; }
      taughtItems.push({ name, vector: normalizedEmbedding(cropBox(teachingTarget)) });
      localStorage.setItem('fieldnote-taught-items', JSON.stringify(taughtItems));
      $('teachDialog').close(); toast(`Saved a view of “${name}” on this device.`);
    } catch (error) { console.error(error); toast('Could not save this view. Browser storage may be full.'); }
  });
  const settingsDialog = $('settingsDialog');
  const confidenceRange = $('confidenceRange'), motionSensitivityRange = $('motionSensitivityRange');
  confidenceRange.value = String(Math.round(confidenceThreshold * 100));
  $('confidenceValue').textContent = `${confidenceRange.value}%`;
  $('customObjectsInput').value = userExtraLabels.join(', ');
  confidenceRange.addEventListener('input', () => {
    confidenceThreshold = Number(confidenceRange.value) / 100;
    $('confidenceValue').textContent = `${confidenceRange.value}%`;
  });
  motionSensitivityRange.addEventListener('input', () => {
    motionThreshold = Number(motionSensitivityRange.value);
    $('motionSensitivityValue').textContent = motionSensitivityRange.value;
  });
  $('settingsButton').addEventListener('click', () => {
    if (typeof settingsDialog.showModal === 'function') settingsDialog.showModal();
    else toast('Settings are not supported by this browser.');
  });
  $('doneSettingsButton').addEventListener('click', () => {
    userExtraLabels = $('customObjectsInput').value.split(',').map(label => label.trim()).filter(Boolean).slice(0, 12);
    try { localStorage.setItem('fieldnote-extra-object-labels', userExtraLabels.join(', ')); }
    catch (error) { console.warn('Could not save custom object labels:', error); }
    if (userExtraLabels.length) toast('Extra object labels saved for Extended mode.');
  });
  $('closeSettingsButton').addEventListener('click', () => settingsDialog.close());
  window.addEventListener('resize', sizeCanvases);
  window.addEventListener('pagehide', () => { running = false; if (stream) stream.getTracks().forEach(t => t.stop()); });
  setInterval(() => { $('cameraTime').textContent = new Date().toLocaleTimeString('en-GB', { hour12: false }); }, 500);
})();
