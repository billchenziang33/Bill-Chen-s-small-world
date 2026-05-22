import { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

const REQUIRED_SAMPLES_PER_DIGIT = 50;
const digitOptions = Array.from({ length: 10 }, (_, index) => String(index + 1));
const objectOptions = [
  { id: "phone", name: "手机" },
  { id: "cup", name: "杯子" },
  { id: "ipad", name: "iPad" },
  { id: "mouse", name: "鼠标" },
  { id: "paper", name: "纸" },
  { id: "tissue", name: "纸巾" }
];
const fixedObjectBox = { x: 0.22, y: 0.2, width: 0.56, height: 0.58 };
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

function apiUrl(path) {
  return `${API_BASE_URL}${path}`;
}

const connections = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17]
];

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const handLandmarkerRef = useRef(null);
  const streamRef = useRef(null);
  const animationFrameRef = useRef(0);
  const lastVideoTimeRef = useRef(-1);
  const latestLandmarksRef = useRef(null);
  const lastSyncedGestureRef = useRef("");
  const isCollectingRef = useRef(false);
  const modeRef = useRef("detect");
  const objectDetectingRef = useRef(false);
  const objectDetectionsRef = useRef([]);

  const [mode, setMode] = useState("detect");
  const [modelStatus, setModelStatus] = useState("正在加载手势模型...");
  const [buttonLabel, setButtonLabel] = useState("开启摄像头");
  const [isReady, setIsReady] = useState(false);
  const [isCameraRunning, setIsCameraRunning] = useState(false);
  const [gestureName, setGestureName] = useState("等待检测");
  const [gestureConfidence, setGestureConfidence] = useState("0%");
  const [recognitionSource, setRecognitionSource] = useState("规则模型");
  const [digitProbabilities, setDigitProbabilities] = useState([]);
  const [selectedDigit, setSelectedDigit] = useState("1");
  const [trainingSamples, setTrainingSamples] = useState([]);
  const [trainedModel, setTrainedModel] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState("选择数字，摆出手势，然后保存样本到本地数据库。");
  const [isCollecting, setIsCollecting] = useState(false);
  const [isTrainingModel, setIsTrainingModel] = useState(false);
  const [selectedObject, setSelectedObject] = useState("phone");
  const [objectNote, setObjectNote] = useState("");
  const [objectSummary, setObjectSummary] = useState({ counts: {}, labels: {}, classes: [], recent: [] });
  const [objectStatus, setObjectStatus] = useState("把物品放进固定蓝色框内，然后保存样本或训练 YOLO。");
  const [isTrainingObjectModel, setIsTrainingObjectModel] = useState(false);
  const [objectDetections, setObjectDetections] = useState([]);
  const [objectProbabilities, setObjectProbabilities] = useState([]);
  const [isObjectDetecting, setIsObjectDetecting] = useState(false);

  const sampleCounts = digitOptions.reduce((counts, digit) => {
    counts[digit] = trainingSamples.filter((sample) => sample.digit === digit).length;
    return counts;
  }, {});
  const selectedDigitCount = sampleCounts[selectedDigit] ?? 0;

  useEffect(() => {
    modeRef.current = mode;
    if (mode === "object") {
      setObjectDetections([]);
      setObjectProbabilities([]);
      objectDetectionsRef.current = [];
      drawObjectOverlay();
    }
  }, [mode]);

  useEffect(() => {
    bootstrap();
    loadTrainingSamples();
    loadTrainedModel();
    loadObjectSummary();
    return () => stopCamera();
  }, []);

  function switchMode(nextMode) {
    setMode(nextMode);
    objectDetectingRef.current = false;
    setIsObjectDetecting(false);
    if (nextMode === "train") setTrainingStatus("选择数字，摆出手势，然后保存样本到本地数据库。");
    if (nextMode === "detect") setTrainingStatus("实时识别会优先展示深度模型 softmax 概率。");
    if (nextMode === "object") setObjectStatus("把物品完整放进固定蓝色框里，再保存样本或开始检测。");
  }

  async function bootstrap() {
    try {
      const vision = await FilesetResolver.forVisionTasks("/mediapipe");
      handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: "/mediapipe/hand_landmarker.task" },
        numHands: 1,
        runningMode: "VIDEO"
      });
      setIsReady(true);
      setModelStatus("模型已就绪");
    } catch (error) {
      console.error(error);
      setModelStatus("手势模型加载失败");
      setGestureName("暂不可用");
    }
  }

  async function loadTrainingSamples() {
    try {
      const response = await fetch(apiUrl("/api/training-samples"));
      const payload = await response.json();
      setTrainingSamples(payload.items || []);
    } catch (error) {
      console.error(error);
      setTrainingStatus("读取本地手势数据库失败，请确认后端正在运行。");
    }
  }

  async function loadTrainedModel() {
    try {
      const response = await fetch(apiUrl("/api/trained-model"));
      const payload = await response.json();
      setTrainedModel(payload.ready ? payload.model : null);
    } catch (error) {
      console.error(error);
      setTrainedModel(null);
    }
  }

  async function loadObjectSummary() {
    try {
      const response = await fetch(apiUrl("/api/object-samples"));
      const payload = await response.json();
      if (!response.ok || payload.error) throw new Error(payload.error || "读取物品样本失败");
      setObjectSummary(payload);
    } catch (error) {
      console.error(error);
      setObjectStatus("读取物品样本失败，请确认后端正在运行。");
    }
  }

  async function toggleCamera() {
    if (!isReady) return;
    if (isCameraRunning) {
      stopCamera();
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setIsCameraRunning(true);
      setButtonLabel("关闭摄像头");
      setModelStatus("摄像头运行中");
      syncCanvasSize();
      renderLoop();
    } catch (error) {
      console.error(error);
      setModelStatus("摄像头权限被拒绝");
      setGestureName("需要摄像头权限");
      setGestureConfidence("0%");
    }
  }

  function stopCamera() {
    isCollectingRef.current = false;
    objectDetectingRef.current = false;
    setIsCollecting(false);
    setIsObjectDetecting(false);
    setIsCameraRunning(false);
    setButtonLabel("开启摄像头");
    setModelStatus((current) => current === "模型已就绪" || current === "手势模型加载失败" ? current : "摄像头已关闭");
    setGestureName((current) => current === "暂不可用" ? current : "已暂停");
    setGestureConfidence("0%");
    setDigitProbabilities([]);
    setObjectDetections([]);
    setObjectProbabilities([]);
    objectDetectionsRef.current = [];
    latestLandmarksRef.current = null;

    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    const canvas = canvasRef.current;
    if (canvas) canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  }

  function renderLoop() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !streamRef.current) return;

    syncCanvasSize();
    if (modeRef.current === "object") {
      drawObjectOverlay();
      animationFrameRef.current = requestAnimationFrame(renderLoop);
      return;
    }

    const handLandmarker = handLandmarkerRef.current;
    if (!handLandmarker) return;
    if (video.currentTime !== lastVideoTimeRef.current) {
      lastVideoTimeRef.current = video.currentTime;
      const results = handLandmarker.detectForVideo(video, performance.now());
      const landmarks = results.landmarks?.[0] ?? null;
      latestLandmarksRef.current = landmarks;
      drawHandResults(results.landmarks ?? []);
      updateGesture(landmarks);
    }
    animationFrameRef.current = requestAnimationFrame(renderLoop);
  }

  function syncCanvasSize() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const width = video.clientWidth;
    const height = video.clientHeight;
    if (!width || !height) return;
    const targetWidth = Math.round(width * dpr);
    const targetHeight = Math.round(height * dpr);
    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
      canvas.width = targetWidth;
      canvas.height = targetHeight;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    }
  }

  function drawHandResults(landmarksList) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    landmarksList.forEach((landmarks) => drawHandSkeleton(context, landmarks));
  }

  function drawObjectOverlay() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    drawNormalizedBox(context, fixedObjectBox, "固定采集框：把物品放进这里", "rgba(0, 166, 214, 0.98)");
    objectDetectionsRef.current.forEach((detection) => {
      drawNormalizedBox(context, detection.box, `${detection.name} ${detection.confidence}%`, "rgba(72, 184, 132, 0.98)");
    });
  }

  function drawHandSkeleton(context, landmarks, color = "rgba(0, 166, 214, 0.96)") {
    const mappedLandmarks = landmarks.map(mapPointToVideoFrame);
    context.lineWidth = 4;
    context.strokeStyle = color;
    context.fillStyle = "rgba(255, 255, 255, 0.95)";
    context.shadowColor = color;
    context.shadowBlur = 8;
    connections.forEach(([start, end]) => {
      const a = mappedLandmarks[start];
      const b = mappedLandmarks[end];
      context.beginPath();
      context.moveTo(a.x, a.y);
      context.lineTo(b.x, b.y);
      context.stroke();
    });
    mappedLandmarks.forEach((point, index) => {
      context.beginPath();
      context.arc(point.x, point.y, [4, 8, 12, 16, 20].includes(index) ? 7 : 4.5, 0, Math.PI * 2);
      context.fill();
    });
    context.shadowBlur = 0;
  }

  function drawNormalizedBox(context, box, label, color) {
    const mapped = mapBoxToVideoFrame(box);
    context.lineWidth = 4;
    context.strokeStyle = color;
    context.fillStyle = color;
    context.shadowColor = color;
    context.shadowBlur = 14;
    context.strokeRect(mapped.x, mapped.y, mapped.width, mapped.height);
    context.shadowBlur = 0;
    context.font = "700 16px Microsoft YaHei, sans-serif";
    const textWidth = context.measureText(label).width;
    context.fillRect(mapped.x, Math.max(0, mapped.y - 34), textWidth + 18, 30);
    context.fillStyle = "white";
    context.fillText(label, mapped.x + 9, Math.max(22, mapped.y - 12));
  }

  function mapPointToVideoFrame(point) {
    const mapped = mapBoxToVideoFrame({ x: point.x, y: point.y, width: 0, height: 0 });
    return { x: mapped.x, y: mapped.y };
  }

  function mapBoxToVideoFrame(box) {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const displayWidth = canvas.clientWidth;
    const displayHeight = canvas.clientHeight;
    const videoWidth = video?.videoWidth || displayWidth;
    const videoHeight = video?.videoHeight || displayHeight;
    const scale = Math.max(displayWidth / videoWidth, displayHeight / videoHeight);
    const renderedWidth = videoWidth * scale;
    const renderedHeight = videoHeight * scale;
    const offsetX = (displayWidth - renderedWidth) / 2;
    const offsetY = (displayHeight - renderedHeight) / 2;
    return {
      x: box.x * renderedWidth + offsetX,
      y: box.y * renderedHeight + offsetY,
      width: box.width * renderedWidth,
      height: box.height * renderedHeight
    };
  }

  function updateGesture(landmarks) {
    if (!landmarks) {
      setGestureName("未检测到手");
      setGestureConfidence("0%");
      setRecognitionSource("等待手势");
      setDigitProbabilities([]);
      return;
    }
    const result = classifyNumberGesture(landmarks);
    const percent = Math.round(result.confidence);
    setGestureName(result.gesture);
    setGestureConfidence(`${percent}%`);
    setRecognitionSource(result.source);
    setDigitProbabilities(result.probabilities || []);
    syncGestureToBackend(result.gesture, percent);
  }

  function classifyNumberGesture(landmarks) {
    const rules = classifyByRules(landmarks);
    const knn = classifyByTraining(landmarks);
    const neural = classifyByNeuralModel(landmarks);
    if (neural && neural.confidence >= 72) {
      return { gesture: `数字 ${neural.digit}`, confidence: neural.confidence, source: "本地深度学习模型", probabilities: neural.probabilities };
    }
    if (knn) {
      const rulesDigit = getDigitFromGesture(rules.gesture);
      if (knn.digit === rulesDigit) {
        return { gesture: `数字 ${knn.digit}`, confidence: Math.min(99, Math.max(knn.confidence, rules.confidence) + 5), source: "样本模型 + 规则融合", probabilities: neural?.probabilities || [] };
      }
      if (knn.confidence >= 74) return { gesture: `数字 ${knn.digit}`, confidence: knn.confidence, source: "本地样本模型", probabilities: neural?.probabilities || [] };
    }
    return { ...rules, source: neural ? "规则模型 + 深度模型参考" : "规则模型", probabilities: neural?.probabilities || [] };
  }

  function classifyByNeuralModel(landmarks) {
    if (!trainedModel?.layers?.length) return null;
    const input = extractFeatureVector(landmarks);
    if (input.length !== trainedModel.featureCount) return null;
    const logits = runMlp(input, trainedModel.layers);
    const probabilities = softmax(logits);
    let bestIndex = 0;
    probabilities.forEach((value, index) => {
      if (value > probabilities[bestIndex]) bestIndex = index;
    });
    return {
      digit: trainedModel.classes[bestIndex],
      confidence: Math.round(probabilities[bestIndex] * 100),
      probabilities: trainedModel.classes.map((digit, index) => ({ digit, probability: Math.round(probabilities[index] * 1000) / 10 }))
    };
  }

  function classifyByTraining(landmarks) {
    if (trainingSamples.length < 20) return null;
    const vector = extractFeatureVector(landmarks);
    const nearest = trainingSamples
      .map((sample) => ({ digit: sample.digit, distance: vectorDistance(vector, sample.vector) }))
      .sort((a, b) => a.distance - b.distance)
      .slice(0, 9);
    if (!nearest.length) return null;
    const scores = {};
    let totalWeight = 0;
    nearest.forEach((item) => {
      const weight = 1 / (item.distance + 0.0001);
      scores[item.digit] = (scores[item.digit] || 0) + weight;
      totalWeight += weight;
    });
    const [digit, score] = Object.entries(scores).sort((a, b) => b[1] - a[1])[0];
    const agreement = score / totalWeight;
    const distanceConfidence = Math.max(0, 1 - nearest[0].distance / 4.2);
    return { digit, confidence: Math.round((agreement * 0.68 + distanceConfidence * 0.32) * 100) };
  }

  function classifyByRules(landmarks) {
    const fingers = getFingerStates(landmarks);
    const openFingers = Object.entries(fingers).filter(([, isOpen]) => isOpen).map(([finger]) => finger);
    const openKey = openFingers.join(",");
    const palmSize = getPalmSize(landmarks);
    const thumbIndexDistance = normalizedDistance(landmarks[4], landmarks[8], palmSize);
    const thumbMiddleDistance = normalizedDistance(landmarks[4], landmarks[12], palmSize);
    const indexMiddleDistance = normalizedDistance(landmarks[8], landmarks[12], palmSize);
    const thumbPinkyDistance = normalizedDistance(landmarks[4], landmarks[20], palmSize);
    const foldedCount = ["index", "middle", "ring", "pinky"].filter((finger) => !fingers[finger]).length;
    const indexCurled = isCurled(landmarks, 8, 6, 5);
    const middleCurled = isCurled(landmarks, 12, 10, 9);
    const ringCurled = isCurled(landmarks, 16, 14, 13);
    const pinkyCurled = isCurled(landmarks, 20, 18, 17);
    if (fingers.thumb && fingers.index && fingers.middle && !fingers.ring && !fingers.pinky && thumbIndexDistance < 0.56 && thumbMiddleDistance < 0.68) return { gesture: "数字 7", confidence: 88 };
    if (fingers.thumb && fingers.index && !fingers.middle && !fingers.ring && !fingers.pinky && thumbIndexDistance > 0.72) return { gesture: "数字 8", confidence: 90 };
    if (fingers.thumb && !fingers.index && !fingers.middle && !fingers.ring && fingers.pinky && thumbPinkyDistance > 1.45) return { gesture: "数字 6", confidence: 92 };
    if (openKey === "index") return { gesture: "数字 1", confidence: 94 };
    if (openKey === "index,middle") return { gesture: "数字 2", confidence: 94 };
    if (fingers.thumb && fingers.index && fingers.middle && !fingers.ring && !fingers.pinky) return { gesture: "数字 3", confidence: indexMiddleDistance < 0.95 ? 90 : 84 };
    if (!fingers.thumb && fingers.index && fingers.middle && fingers.ring && fingers.pinky) return { gesture: "数字 4", confidence: 94 };
    if (fingers.index && fingers.middle && fingers.ring && fingers.pinky && foldedCount <= 1) return { gesture: "数字 4", confidence: 86 };
    if (openFingers.length === 5) return { gesture: "数字 5", confidence: 96 };
    if (fingers.thumb && fingers.pinky && middleCurled && ringCurled && !fingers.index) return { gesture: "数字 6", confidence: 84 };
    if (fingers.thumb && fingers.index && middleCurled && ringCurled && pinkyCurled) return { gesture: "数字 8", confidence: 84 };
    if (!fingers.thumb && indexCurled && !fingers.middle && !fingers.ring && !fingers.pinky) return { gesture: "数字 9", confidence: 76 };
    if (openFingers.length === 0) return { gesture: "数字 10", confidence: 90 };
    return { gesture: "未匹配数字", confidence: 45 };
  }

  async function captureTrainingSample() {
    const landmarks = latestLandmarksRef.current;
    if (!landmarks) {
      setTrainingStatus("还没有检测到手，请先把手放进画面。");
      return false;
    }
    const sample = { id: crypto.randomUUID(), digit: selectedDigit, vector: extractFeatureVector(landmarks) };
    try {
      await fetch(apiUrl("/api/training-samples"), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(sample) });
      await loadTrainingSamples();
      setTrainingStatus(`已保存数字 ${selectedDigit} 的样本到本地数据库。`);
      return true;
    } catch (error) {
      console.error(error);
      setTrainingStatus("保存样本失败，请确认后端正在运行。");
      return false;
    }
  }

  async function collectUntilReady() {
    if (!isCameraRunning) {
      setTrainingStatus("请先开启摄像头，再连续采集样本。");
      return;
    }
    isCollectingRef.current = true;
    setIsCollecting(true);
    let collected = selectedDigitCount;
    while (isCollectingRef.current && collected < REQUIRED_SAMPLES_PER_DIGIT) {
      const saved = await captureTrainingSample();
      if (saved) collected += 1;
      setTrainingStatus(`正在保存数字 ${selectedDigit}：${collected}/${REQUIRED_SAMPLES_PER_DIGIT}`);
      await wait(70);
    }
    isCollectingRef.current = false;
    setIsCollecting(false);
    setTrainingStatus(collected >= REQUIRED_SAMPLES_PER_DIGIT ? `数字 ${selectedDigit} 已采满 50 个样本。` : `已停止采集数字 ${selectedDigit}。`);
  }

  async function trainDeepModel() {
    setIsTrainingModel(true);
    setTrainingStatus("深度学习模型训练已提交到云端后台，请稍等。");
    try {
      const response = await fetch(apiUrl("/api/train-model"), { method: "POST" });
      const payload = await response.json();
      if (!response.ok || payload.error) throw new Error(payload.error || "训练失败");
      await pollGestureTrainingStatus();
    } catch (error) {
      console.error(error);
      setTrainingStatus(error.message);
      setIsTrainingModel(false);
    }
  }

  async function pollGestureTrainingStatus() {
    for (let attempt = 0; attempt < 180; attempt += 1) {
      const response = await fetch(apiUrl("/api/train-model-status"));
      const payload = await response.json();
      const job = payload.job || {};

      if (job.status === "completed") {
        await loadTrainedModel();
        const result = job.payload || {};
        const accuracy = result.accuracy ? (result.accuracy * 100).toFixed(1) : "未知";
        setTrainingStatus(`训练完成：${result.sampleCount ?? "若干"} 个样本，训练集准确率 ${accuracy}%。`);
        setIsTrainingModel(false);
        return;
      }

      if (job.status === "failed") {
        throw new Error(job.error || "训练失败");
      }

      setTrainingStatus(`深度学习模型正在云端后台训练中... ${attempt + 1}`);
      await wait(2000);
    }

      await loadTrainedModel();
      setTrainingStatus("训练仍在后台运行，可以稍后刷新页面查看模型状态。");
      setIsTrainingModel(false);
  }

  function stopCollecting() {
    isCollectingRef.current = false;
    setIsCollecting(false);
  }

  async function clearSelectedDigitSamples() {
    await fetch(apiUrl("/api/training-samples/clear"), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ digit: selectedDigit }) });
    await loadTrainingSamples();
    setTrainingStatus(`已清空数字 ${selectedDigit} 的数据库样本。`);
  }

  function captureFrameDataUrl() {
    const video = videoRef.current;
    if (!video || !video.videoWidth || !video.videoHeight) throw new Error("请先开启摄像头。");
    const frameCanvas = document.createElement("canvas");
    frameCanvas.width = video.videoWidth;
    frameCanvas.height = video.videoHeight;
    frameCanvas.getContext("2d").drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);
    return frameCanvas.toDataURL("image/jpeg", 0.86);
  }

  async function saveObjectSample() {
    try {
      const image = captureFrameDataUrl();
      const response = await fetch(apiUrl("/api/object-samples"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label: selectedObject, image, box: fixedObjectBox, note: objectNote })
      });
      const payload = await response.json();
      if (!response.ok || payload.error) throw new Error(payload.error || "保存失败");
      setObjectSummary(payload.summary);
      const label = objectOptions.find((item) => item.id === selectedObject)?.name || selectedObject;
      setObjectStatus(`已保存 1 张 ${label} 固定框样本。`);
    } catch (error) {
      console.error(error);
      setObjectStatus(error.message);
    }
  }

  async function trainObjectModel() {
    setIsTrainingObjectModel(true);
    setObjectStatus("YOLO 训练已提交到云端后台，请稍等。");
    try {
      const response = await fetch(apiUrl("/api/train-object-model"), { method: "POST" });
      const payload = await response.json();
      if (!response.ok || payload.error) throw new Error(payload.error || "YOLO 训练失败");
      await pollObjectTrainingStatus();
    } catch (error) {
      console.error(error);
      setObjectStatus(error.message);
      setIsTrainingObjectModel(false);
    }
  }

  async function pollObjectTrainingStatus() {
    for (let attempt = 0; attempt < 240; attempt += 1) {
      const response = await fetch(apiUrl("/api/object-train-status"));
      const payload = await response.json();
      const job = payload.job || {};

      if (job.status === "completed") {
        const epochs = job.payload?.epochs ?? "若干";
        setObjectStatus(`YOLO 训练完成：${epochs} 轮，模型已保存。`);
        setIsTrainingObjectModel(false);
        return;
      }

      if (job.status === "failed") {
        throw new Error(job.error || "YOLO 训练失败");
      }

      setObjectStatus(`YOLO 正在云端后台训练中... ${attempt + 1}`);
      await delay(3000);
    }

    setObjectStatus("YOLO 训练仍在后台运行，可以稍后刷新页面查看。");
    setIsTrainingObjectModel(false);
  }

  async function detectObjectsOnce() {
    try {
      const image = captureFrameDataUrl();
      const response = await fetch(apiUrl("/api/object-detect"), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image }) });
      const payload = await response.json();
      if (!response.ok || payload.error) throw new Error(payload.error || "检测失败");
      const detections = payload.detections || [];
      const probabilities = payload.probabilities || [];
      setObjectDetections(detections);
      setObjectProbabilities(probabilities);
      objectDetectionsRef.current = detections;
      drawObjectOverlay();
      const best = probabilities[0];
      setObjectStatus(best?.probability > 0 ? `当前最可能是：${best.name}，概率 ${best.probability}%。` : "这一帧没有检测到 YOLO 目标。");
    } catch (error) {
      console.error(error);
      setObjectStatus(error.message);
    }
  }

  async function toggleObjectDetectionLoop() {
    if (!isCameraRunning) {
      setObjectStatus("请先开启摄像头，再开始实时物品检测。");
      return;
    }
    if (objectDetectingRef.current) {
      objectDetectingRef.current = false;
      setIsObjectDetecting(false);
      setObjectStatus("已停止实时物品检测。");
      return;
    }
    objectDetectingRef.current = true;
    setIsObjectDetecting(true);
    setObjectStatus("正在实时检测固定框内物品...");
    while (objectDetectingRef.current) {
      await detectObjectsOnce();
      await wait(900);
    }
  }

  async function clearSelectedObjectSamples() {
    await fetch(apiUrl("/api/object-samples/clear"), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ label: selectedObject }) });
    await loadObjectSummary();
    const label = objectOptions.find((item) => item.id === selectedObject)?.name || selectedObject;
    setObjectStatus(`已清空 ${label} 的物品样本。`);
  }

  function extractFeatureVector(landmarks) {
    const palmSize = getPalmSize(landmarks);
    const palmCenter = getPalmCenter(landmarks);
    const handSign = landmarks[17].x < landmarks[5].x ? 1 : -1;
    const relativeLandmarks = landmarks.flatMap((point) => [((point.x - palmCenter.x) * handSign) / palmSize, (point.y - palmCenter.y) / palmSize, (point.z - palmCenter.z) / palmSize]);
    const fingers = getFingerStates(landmarks);
    const fingerVector = ["thumb", "index", "middle", "ring", "pinky"].map((finger) => fingers[finger] ? 1 : 0);
    return [...relativeLandmarks, ...fingerVector];
  }

  function getFingerStates(landmarks) {
    const palmSize = getPalmSize(landmarks);
    const palmCenter = getPalmCenter(landmarks);
    const thumbTipDistance = normalizedDistance(landmarks[4], palmCenter, palmSize);
    const thumbIpDistance = normalizedDistance(landmarks[3], palmCenter, palmSize);
    const thumbMcpDistance = normalizedDistance(landmarks[2], palmCenter, palmSize);
    const thumbTipSpread = normalizedDistance(landmarks[4], landmarks[5], palmSize);
    return {
      thumb: thumbTipDistance > thumbIpDistance * 1.08 && thumbTipDistance > thumbMcpDistance * 1.18 && thumbTipSpread > 0.72,
      index: isFingerExtended(landmarks, 8, 6, 5),
      middle: isFingerExtended(landmarks, 12, 10, 9),
      ring: isFingerExtended(landmarks, 16, 14, 13),
      pinky: isFingerExtended(landmarks, 20, 18, 17)
    };
  }

  function isFingerExtended(landmarks, tipIndex, pipIndex, mcpIndex) {
    const palmSize = getPalmSize(landmarks);
    const wrist = landmarks[0];
    const tipToWrist = normalizedDistance(landmarks[tipIndex], wrist, palmSize);
    const pipToWrist = normalizedDistance(landmarks[pipIndex], wrist, palmSize);
    const mcpToWrist = normalizedDistance(landmarks[mcpIndex], wrist, palmSize);
    return tipToWrist > pipToWrist * 1.08 && tipToWrist > mcpToWrist * 1.34 && landmarks[tipIndex].y < landmarks[pipIndex].y + palmSize * 0.12;
  }

  function isCurled(landmarks, tipIndex, pipIndex, mcpIndex) {
    const palmSize = getPalmSize(landmarks);
    const tipToMcp = normalizedDistance(landmarks[tipIndex], landmarks[mcpIndex], palmSize);
    const pipToMcp = normalizedDistance(landmarks[pipIndex], landmarks[mcpIndex], palmSize);
    return tipToMcp < pipToMcp * 1.22 || landmarks[tipIndex].y > landmarks[pipIndex].y;
  }

  function getPalmCenter(landmarks) {
    return {
      x: (landmarks[0].x + landmarks[5].x + landmarks[9].x + landmarks[13].x + landmarks[17].x) / 5,
      y: (landmarks[0].y + landmarks[5].y + landmarks[9].y + landmarks[13].y + landmarks[17].y) / 5,
      z: (landmarks[0].z + landmarks[5].z + landmarks[9].z + landmarks[13].z + landmarks[17].z) / 5
    };
  }

  function getPalmSize(landmarks) {
    return Math.max(distance(landmarks[0], landmarks[9]), distance(landmarks[5], landmarks[17]), 0.001);
  }

  function normalizedDistance(a, b, scale) {
    return distance(a, b) / scale;
  }

  function distance(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
  }

  function vectorDistance(a, b) {
    let sum = 0;
    for (let index = 0; index < a.length; index += 1) {
      const diff = a[index] - b[index];
      sum += diff * diff;
    }
    return Math.sqrt(sum / a.length);
  }

  function getDigitFromGesture(gesture) {
    const match = gesture.match(/\d+/);
    return match ? match[0] : "";
  }

  function runMlp(input, layers) {
    return layers.reduce((values, layer) => {
      const output = layer.weight.map((weights, rowIndex) => weights.reduce((sum, weight, columnIndex) => sum + weight * values[columnIndex], layer.bias[rowIndex]));
      return layer.activation === "relu" ? output.map((value) => Math.max(0, value)) : output;
    }, input);
  }

  function softmax(values) {
    const max = Math.max(...values);
    const exps = values.map((value) => Math.exp(value - max));
    const sum = exps.reduce((total, value) => total + value, 0);
    return exps.map((value) => value / sum);
  }

  async function syncGestureToBackend(gesture, confidence) {
    const signature = `${gesture}:${confidence}`;
    if (gesture === "未匹配数字" || gesture === "未检测到手" || signature === lastSyncedGestureRef.current) return;
    lastSyncedGestureRef.current = signature;
    try {
        await fetch(apiUrl("/api/gestures"), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ gesture, confidence }) });
    } catch (error) {
      console.error(error);
    }
  }

  return (
    <main className="page-shell">
      <section className="app-layout">
        <header className="top-bar">
          <div>
            <p className="eyebrow">Bill Chen 的小世界</p>
            <h1>视觉识别实验室</h1>
          </div>
          <div className="mode-switch">
            <button className={mode === "train" ? "mode-button active" : "mode-button"} type="button" onClick={() => switchMode("train")}>手势训练</button>
            <button className={mode === "detect" ? "mode-button active" : "mode-button"} type="button" onClick={() => switchMode("detect")}>手势识别</button>
            <button className={mode === "object" ? "mode-button active" : "mode-button"} type="button" onClick={() => switchMode("object")}>物品检测</button>
          </div>
        </header>

        <section className="workspace-grid">
          <div className="camera-panel">
            <div className="video-frame">
              <video ref={videoRef} autoPlay playsInline muted id="camera-feed" />
              <canvas ref={canvasRef} id="gesture-overlay" />
              <div className="frame-glow" />
            </div>
            <div className="action-row">
              <button className="primary-button" type="button" onClick={toggleCamera} disabled={!isReady}>{buttonLabel}</button>
              <span className="status-pill">{modelStatus}</span>
            </div>
          </div>

          <div className="control-panel">
            {mode === "train" && (
              <>
                <div className="panel-header">
                  <p className="label">手势训练入口</p>
                  <h2>给手势绑定数字</h2>
                  <p>选择数字，摆出对应手势，把关键点样本保存到本地 SQLite 数据库；采满后训练本地深度学习模型。</p>
                </div>
                <div className="training-panel">
                  <div className="training-header">
                    <div><p className="label">当前标签</p><strong>数字 {selectedDigit}：{selectedDigitCount}/{REQUIRED_SAMPLES_PER_DIGIT}</strong></div>
                    <select value={selectedDigit} onChange={(event) => setSelectedDigit(event.target.value)}>
                      {digitOptions.map((digit) => <option value={digit} key={digit}>数字 {digit}</option>)}
                    </select>
                  </div>
                  <div className="training-actions">
                    <button className="ghost-button" type="button" onClick={captureTrainingSample}>保存 1 个样本</button>
                    <button className="ghost-button" type="button" onClick={collectUntilReady} disabled={isCollecting}>连续保存到 100</button>
                    <button className="ghost-button" type="button" onClick={stopCollecting} disabled={!isCollecting}>停止</button>
                    <button className="ghost-button" type="button" onClick={trainDeepModel} disabled={isTrainingModel}>{isTrainingModel ? "训练中..." : "训练深度模型"}</button>
                    <button className="ghost-button danger-button" type="button" onClick={clearSelectedDigitSamples}>清空当前数字</button>
                  </div>
                  <p className="training-status">{trainingStatus}</p>
                  <div className="sample-grid">
                    {digitOptions.map((digit) => <span className={(sampleCounts[digit] ?? 0) >= REQUIRED_SAMPLES_PER_DIGIT ? "sample-ready" : ""} key={digit}>{digit}: {sampleCounts[digit] ?? 0}</span>)}
                  </div>
                </div>
              </>
            )}

            {mode === "detect" && (
              <>
                <div className="panel-header">
                  <p className="label">实时手势识别入口</p>
                  <h2>{gestureName}</h2>
                  <p>根据当前手势实时输出 softmax 概率，展示这个手势分别对应数字 1 到 10 的可能性。</p>
                </div>
                <div className="gesture-summary">
                  <div><p className="label">置信度</p><h3>{gestureConfidence}</h3></div>
                  <div><p className="label">识别来源</p><h3>{recognitionSource}</h3></div>
                </div>
                <div className="probability-panel">
                  <p className="label">Softmax 数字概率</p>
                  <div className="probability-list">
                    {digitOptions.map((digit) => {
                      const item = digitProbabilities.find((probability) => probability.digit === digit);
                      const probability = item?.probability ?? 0;
                      return <div className="probability-row" key={digit}><span>数字 {digit}</span><div className="probability-track"><i style={{ width: `${probability}%` }} /></div><strong>{probability.toFixed(1)}%</strong></div>;
                    })}
                  </div>
                </div>
              </>
            )}

            {mode === "object" && (
              <>
                <div className="panel-header">
                  <p className="label">固定框物品检测入口</p>
                  <h2>采集、备注、训练 YOLO</h2>
                  <p>把物品放进固定蓝色框内，选择类别并保存样本。训练 YOLO 后，概率面板会显示框内物品属于每个类别的可能性。</p>
                </div>
                <div className="training-panel">
                  <div className="training-header">
                    <div><p className="label">当前物品</p><strong>{objectOptions.find((item) => item.id === selectedObject)?.name}</strong></div>
                    <select value={selectedObject} onChange={(event) => setSelectedObject(event.target.value)}>
                      {objectOptions.map((item) => <option value={item.id} key={item.id}>{item.name}</option>)}
                    </select>
                  </div>
                  <label className="note-field">
                    <span>备注当前框内的物品</span>
                    <textarea value={objectNote} onChange={(event) => setObjectNote(event.target.value)} placeholder="比如：黑色手机、透明玻璃杯、白色纸巾包..." />
                  </label>
                  <div className="training-actions">
                    <button className="ghost-button" type="button" onClick={saveObjectSample}>保存固定框样本</button>
                    <button className="ghost-button" type="button" onClick={trainObjectModel} disabled={isTrainingObjectModel}>{isTrainingObjectModel ? "YOLO 训练中..." : "训练 YOLO 模型"}</button>
                    <button className="ghost-button" type="button" onClick={detectObjectsOnce}>YOLO 检测一次</button>
                    <button className="ghost-button" type="button" onClick={toggleObjectDetectionLoop}>{isObjectDetecting ? "停止实时检测" : "开始实时检测"}</button>
                    <button className="ghost-button danger-button" type="button" onClick={clearSelectedObjectSamples}>清空当前物品</button>
                  </div>
                  <p className="training-status">{objectStatus}</p>
                  <div className="sample-grid object-grid">
                    {objectOptions.map((item) => <span className={(objectSummary.counts?.[item.id] ?? 0) >= 30 ? "sample-ready" : ""} key={item.id}>{item.name}: {objectSummary.counts?.[item.id] ?? 0}</span>)}
                  </div>
                </div>
                <div className="probability-panel">
                  <p className="label">框内物品类别概率</p>
                  <div className="probability-list">
                    {objectOptions.map((option) => {
                      const item = objectProbabilities.find((probability) => probability.label === option.id);
                      const probability = item?.probability ?? 0;
                      return <div className="probability-row" key={option.id}><span>{option.name}</span><div className="probability-track"><i style={{ width: `${probability}%` }} /></div><strong>{probability.toFixed(1)}%</strong></div>;
                    })}
                  </div>
                </div>
                <div className="probability-panel">
                  <p className="label">检测结果与最近备注</p>
                  <div className="object-detection-list">
                    {objectDetections.length === 0 ? <span>还没有 YOLO 检测结果。请先训练模型，或点击检测一次。</span> : objectDetections.map((item, index) => <strong key={`${item.label}-${index}`}>{item.name}：{item.confidence}%</strong>)}
                    {(objectSummary.recent || []).slice(0, 4).map((item) => <span key={item.id}>{item.name}：{item.note || "无备注"}</span>)}
                  </div>
                </div>
              </>
            )}
          </div>
        </section>
      </section>
    </main>
  );
}

function wait(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}
