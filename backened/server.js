const http = require("http");
const fs = require("fs");
const path = require("path");
const { spawn, spawnSync } = require("child_process");

const port = process.env.PORT || 3000;
const host = process.env.HOST || "0.0.0.0";
const rootDir = path.resolve(__dirname, "..");
const frontendDistDir = path.join(rootDir, "frontened", "dist");
const backendDir = __dirname;
const trainingDbScript = path.join(backendDir, "training_db.py");
const trainModelScript = path.join(backendDir, "ml", "train_model.py");
const trainedModelPath = path.join(backendDir, "data", "gesture_model.json");
const objectDatasetScript = path.join(backendDir, "object_dataset.py");
const trainYoloScript = path.join(backendDir, "ml", "train_yolo_objects.py");
const detectYoloScript = path.join(backendDir, "ml", "detect_yolo_objects.py");
const objectTempDir = path.join(backendDir, "data", "object_temp");

const mimeTypes = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon"
};

const gestureHistory = [];
const maxHistorySize = 12;
const objectTrainingJob = {
  status: "idle",
  startedAt: null,
  finishedAt: null,
  payload: null,
  error: null,
  logs: []
};
const gestureTrainingJob = {
  status: "idle",
  startedAt: null,
  finishedAt: null,
  payload: null,
  error: null,
  logs: []
};

function sendJson(response, statusCode, payload) {
  response.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Access-Control-Allow-Origin": "*"
  });
  response.end(JSON.stringify(payload));
}

function serveFile(response, filePath) {
  fs.readFile(filePath, (error, content) => {
    if (error) {
      sendJson(response, 404, { error: "File not found" });
      return;
    }

    const extension = path.extname(filePath).toLowerCase();
    response.writeHead(200, {
      "Content-Type": mimeTypes[extension] || "application/octet-stream"
    });
    response.end(content);
  });
}

function collectRequestBody(request) {
  return new Promise((resolve, reject) => {
    let rawBody = "";

    request.on("data", (chunk) => {
      rawBody += chunk;
      if (rawBody.length > 20e6) {
        reject(new Error("Payload too large"));
        request.destroy();
      }
    });

    request.on("end", () => {
      try {
        resolve(rawBody ? JSON.parse(rawBody) : {});
      } catch {
        reject(new Error("Invalid JSON body"));
      }
    });

    request.on("error", reject);
  });
}

function runPythonJson(args, input = null) {
  const result = spawnSync("python", args, {
    cwd: rootDir,
    env: { ...process.env, PYTHONIOENCODING: "utf-8" },
    encoding: "utf-8",
    input
  });

  if (result.error) {
    throw result.error;
  }

  const output = (result.stdout || "").trim();
  const errorOutput = (result.stderr || "").trim();
  let payload = {};

  try {
    if (!output) {
      payload = {};
    } else {
      const jsonLine = output
        .split(/\r?\n/)
        .reverse()
        .find((line) => line.trim().startsWith("{") && line.trim().endsWith("}"));
      payload = JSON.parse(jsonLine || output);
    }
  } catch {
    payload = { error: errorOutput || "Python command failed" };
  }

  if (result.status !== 0 || payload.error) {
    throw new Error(payload.error || errorOutput || "Python command failed");
  }

  return payload;
}

function parsePythonJsonOutput(output, errorOutput = "") {
  if (!output.trim()) {
    return {};
  }

  const jsonLine = output
    .trim()
    .split(/\r?\n/)
    .reverse()
    .find((line) => line.trim().startsWith("{") && line.trim().endsWith("}"));

  try {
    return JSON.parse(jsonLine || output.trim());
  } catch {
    return { error: errorOutput || "Python command failed" };
  }
}

function startPythonTrainingJob(job, scriptPath) {
  if (job.status === "running") {
    return { started: false, job };
  }

  job.status = "running";
  job.startedAt = new Date().toISOString();
  job.finishedAt = null;
  job.payload = null;
  job.error = null;
  job.logs = [];

  const child = spawn("python", [scriptPath], {
    cwd: rootDir,
    env: { ...process.env, PYTHONIOENCODING: "utf-8" },
    windowsHide: true
  });

  let stdout = "";
  let stderr = "";

  child.stdout.on("data", (chunk) => {
    const text = chunk.toString("utf-8");
    stdout += text;
    job.logs.push(...text.trim().split(/\r?\n/).filter(Boolean).slice(-10));
    job.logs = job.logs.slice(-30);
  });

  child.stderr.on("data", (chunk) => {
    const text = chunk.toString("utf-8");
    stderr += text;
    job.logs.push(...text.trim().split(/\r?\n/).filter(Boolean).slice(-10));
    job.logs = job.logs.slice(-30);
  });

  child.on("error", (error) => {
    job.status = "failed";
    job.error = error.message;
    job.finishedAt = new Date().toISOString();
  });

  child.on("close", (code) => {
    const payload = parsePythonJsonOutput(stdout, stderr);
    job.finishedAt = new Date().toISOString();

    if (code === 0 && !payload.error) {
      job.status = "completed";
      job.payload = payload;
      job.error = null;
      return;
    }

    job.status = "failed";
    job.payload = payload;
    job.error = payload.error || stderr.trim() || `Python exited with code ${code}`;
  });

  return { started: true, job };
}

function startGestureTrainingJob() {
  return startPythonTrainingJob(gestureTrainingJob, trainModelScript);
}

function startObjectTrainingJob() {
  return startPythonTrainingJob(objectTrainingJob, trainYoloScript);
}

function readTrainedModel() {
  if (!fs.existsSync(trainedModelPath)) {
    return null;
  }

  return JSON.parse(fs.readFileSync(trainedModelPath, "utf-8"));
}

function saveDataUrlImage(dataUrl) {
  if (typeof dataUrl !== "string" || !dataUrl.includes(",")) {
    throw new Error("image must be a data URL");
  }

  fs.mkdirSync(objectTempDir, { recursive: true });
  const imageBuffer = Buffer.from(dataUrl.split(",", 2)[1], "base64");
  const imagePath = path.join(objectTempDir, `detect_${Date.now()}.jpg`);
  fs.writeFileSync(imagePath, imageBuffer);
  return imagePath;
}

const server = http.createServer(async (request, response) => {
  const requestUrl = new URL(request.url, `http://${request.headers.host}`);
  const pathname = requestUrl.pathname;

  if (request.method === "OPTIONS") {
    response.writeHead(204, {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type"
    });
    response.end();
    return;
  }

  if (pathname === "/api/health" && request.method === "GET") {
    sendJson(response, 200, {
      ok: true,
      service: "gesture-api",
      timestamp: new Date().toISOString(),
      trackedGestures: gestureHistory.length
      ,
      gestureTraining: gestureTrainingJob.status,
      objectTraining: objectTrainingJob.status
    });
    return;
  }

  if (pathname === "/api/gestures" && request.method === "GET") {
    sendJson(response, 200, {
      items: gestureHistory
    });
    return;
  }

  if (pathname === "/api/gestures" && request.method === "POST") {
    try {
      const body = await collectRequestBody(request);
      const gesture = typeof body.gesture === "string" ? body.gesture.trim() : "";
      const confidence = Number(body.confidence) || 0;

      if (!gesture) {
        sendJson(response, 400, { error: "gesture is required" });
        return;
      }

      const entry = {
        id: Date.now().toString(36),
        gesture,
        confidence: Math.max(0, Math.min(100, Math.round(confidence))),
        timestamp: new Date().toISOString()
      };

      gestureHistory.unshift(entry);
      if (gestureHistory.length > maxHistorySize) {
        gestureHistory.pop();
      }

      sendJson(response, 201, {
        saved: true,
        entry
      });
    } catch (error) {
      sendJson(response, 400, { error: error.message });
    }
    return;
  }

  if (pathname === "/api/training-samples" && request.method === "GET") {
    try {
      sendJson(response, 200, runPythonJson([trainingDbScript, "list"]));
    } catch (error) {
      sendJson(response, 500, { error: error.message });
    }
    return;
  }

  if (pathname === "/api/training-samples" && request.method === "POST") {
    try {
      const body = await collectRequestBody(request);
      sendJson(response, 201, runPythonJson([trainingDbScript, "add", JSON.stringify(body)]));
    } catch (error) {
      sendJson(response, 400, { error: error.message });
    }
    return;
  }

  if (pathname === "/api/training-samples/clear" && request.method === "POST") {
    try {
      const body = await collectRequestBody(request);
      sendJson(response, 200, runPythonJson([trainingDbScript, "clear", JSON.stringify(body)]));
    } catch (error) {
      sendJson(response, 400, { error: error.message });
    }
    return;
  }

  if (pathname === "/api/train-model" && request.method === "POST") {
    try {
      const result = startGestureTrainingJob();
      sendJson(response, result.started ? 202 : 200, result);
    } catch (error) {
      sendJson(response, 500, { error: error.message });
    }
    return;
  }

  if (pathname === "/api/train-model-status" && request.method === "GET") {
    sendJson(response, 200, { job: gestureTrainingJob });
    return;
  }

  if (pathname === "/api/trained-model" && request.method === "GET") {
    try {
      const model = readTrainedModel();
      sendJson(response, 200, {
        model,
        ready: Boolean(model)
      });
    } catch (error) {
      sendJson(response, 500, { error: error.message });
    }
    return;
  }

  if (pathname === "/api/object-samples" && request.method === "GET") {
    try {
      sendJson(response, 200, runPythonJson([objectDatasetScript, "summary"]));
    } catch (error) {
      sendJson(response, 500, { error: error.message });
    }
    return;
  }

  if (pathname === "/api/object-samples" && request.method === "POST") {
    try {
      const body = await collectRequestBody(request);
      sendJson(response, 201, runPythonJson([objectDatasetScript, "add", "-"], JSON.stringify(body)));
    } catch (error) {
      sendJson(response, 400, { error: error.message });
    }
    return;
  }

  if (pathname === "/api/object-samples/clear" && request.method === "POST") {
    try {
      const body = await collectRequestBody(request);
      sendJson(response, 200, runPythonJson([objectDatasetScript, "clear", JSON.stringify(body)]));
    } catch (error) {
      sendJson(response, 400, { error: error.message });
    }
    return;
  }

  if (pathname === "/api/train-object-model" && request.method === "POST") {
    try {
      const result = startObjectTrainingJob();
      sendJson(response, result.started ? 202 : 200, result);
    } catch (error) {
      sendJson(response, 500, { error: error.message });
    }
    return;
  }

  if (pathname === "/api/object-train-status" && request.method === "GET") {
    sendJson(response, 200, { job: objectTrainingJob });
    return;
  }

  if (pathname === "/api/object-detect" && request.method === "POST") {
    let imagePath = "";
    try {
      const body = await collectRequestBody(request);
      imagePath = saveDataUrlImage(body.image);
      sendJson(response, 200, runPythonJson([detectYoloScript, imagePath]));
    } catch (error) {
      sendJson(response, 500, { error: error.message });
    } finally {
      if (imagePath) fs.rm(imagePath, { force: true }, () => {});
    }
    return;
  }

  const normalizedPath = pathname === "/" ? "index.html" : pathname.replace(/^\/+/, "");
  const filePath = path.resolve(frontendDistDir, normalizedPath);

  if (!filePath.startsWith(frontendDistDir)) {
    sendJson(response, 403, { error: "Forbidden" });
    return;
  }

  fs.access(filePath, fs.constants.F_OK, (error) => {
    if (!error) {
      serveFile(response, filePath);
      return;
    }

    const fallbackFile = path.join(frontendDistDir, "index.html");
    fs.access(fallbackFile, fs.constants.F_OK, (fallbackError) => {
      if (fallbackError) {
        sendJson(response, 404, {
          error: "Frontend build not found. Run npm run build or npm run dev first."
        });
        return;
      }

      serveFile(response, fallbackFile);
    });
  });
});

server.listen(port, host, () => {
  console.log(`Bill Chen's Small World is running at http://${host}:${port}`);
});
