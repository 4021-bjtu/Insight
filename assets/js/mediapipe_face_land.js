import vision from "./mediapipe_tasks-vision@0.10.3.js"; // Adjust path if needed
const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;

const demosSection = document.getElementById("demos");
const videoBlendShapes = document.getElementById("video-blend-shapes");
const gazePointElement = document.getElementById("gazePoint");
const calibrationPointElement = document.getElementById("calibrationPoint");
const calibrationInstructionElement = document.getElementById("calibrationInstruction");

let faceLandmarker;
let runningMode = "IMAGE";
let enableWebcamButton;
let calibrateButton;
let clearCalibrationButton;
let webcamRunning = false;
const videoWidth = 480;

// --- Gaze Calculation Variables ---
let smoothedGazeX = window.innerWidth / 2;
let smoothedGazeY = window.innerHeight / 2;
const SMOOTHING_FACTOR = 0.5;

// Default gaze parameters (will be overwritten by calibration)
let calibratedGazeParams = {
    minRelX: 0.35, 
    maxRelX: 0.65, 
    minRelY: 0.35, 
    maxRelY: 0.65, 
    invertX: true, // Assume video is mirrored or interpretation requires inversion
    invertY: false 
};

// --- Calibration Variables ---
let isCalibrating = false;
let calibrationState = "idle"; // 'idle', 'waiting_for_input', 'collecting_samples'
let currentCalibrationPointIndex = 0;
const CALIBRATION_SAMPLES_PER_POINT = 60; // Number of frames to average for each point
let samplesCollectedThisPoint = 0;
let accumulatedRelX = 0;
let accumulatedRelY = 0;
let calibrationDataEye = []; // Stores {avgRelX, avgRelY} for each screen point

// Screen points for calibration (normalized 0-1, with a small inset)
const CALIBRATION_POINT_SCREEN_COORDS = [
    { x: 0.05, y: 0.05 }, // Top-Left
    { x: 0.95, y: 0.05 }, // Top-Right
    { x: 0.95, y: 0.95 }, // Bottom-Right
    { x: 0.05, y: 0.95 }  // Bottom-Left
];

async function createFaceLandmarker() {
    const filesetResolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
            delegate: "GPU"
        },
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
        runningMode,
        numFaces: 1
    });
    demosSection.classList.remove("invisible");
    console.log("FaceLandmarker created successfully");
    loadCalibrationData(); // Load any saved calibration
}
createFaceLandmarker();

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

function hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    calibrateButton = document.getElementById("calibrateButton");
    clearCalibrationButton = document.getElementById("clearCalibrationButton");

    enableWebcamButton.addEventListener("click", enableCam);
    calibrateButton.addEventListener("click", startCalibration);
    clearCalibrationButton.addEventListener("click", clearCalibration);
    window.addEventListener("keydown", handleKeyPress); // For calibration input
} else {
    console.warn("getUserMedia() is not supported by your browser");
}

function enableCam(event) {
    if (!faceLandmarker) {
        console.log("Wait! faceLandmarker not loaded yet.");
        return;
    }
    if (webcamRunning === true) {
        webcamRunning = false;
        enableWebcamButton.innerText = "ENABLE WEBCAM";
        calibrateButton.style.display = "none";
        clearCalibrationButton.style.display = "none";
        video.pause();
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
            video.srcObject = null;
        }
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        gazePointElement.style.display = "none";
        stopCalibration(); // Ensure calibration UI is hidden
    } else {
        webcamRunning = true;
        enableWebcamButton.innerText = "DISABLE WEBCAM";
        calibrateButton.style.display = "inline-block";
        clearCalibrationButton.style.display = "inline-block";
        const constraints = { video: true };
        navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
            video.srcObject = stream;
            video.addEventListener("loadeddata", predictWebcam);
        }).catch(err => {
            console.error("Error accessing webcam: ", err);
            webcamRunning = false;
            enableWebcamButton.innerText = "ENABLE WEBCAM";
            calibrateButton.style.display = "none";
            clearCalibrationButton.style.display = "none";
        });
    }
}

function getUniqueLandmarkIndices(connectorConstant) {
    if (!connectorConstant) return [];
    return [...new Set(connectorConstant.flatMap(conn => [conn.start, conn.end]))];
}

function getAveragePosition(landmarks, indices) {
    if (!landmarks || indices.length === 0) return null;
    let sumX = 0, sumY = 0, sumZ = 0;
    let count = 0;
    for (const index of indices) {
        if (landmarks[index]) {
            sumX += landmarks[index].x;
            sumY += landmarks[index].y;
            sumZ += landmarks[index].z;
            count++;
        }
    }
    return count > 0 ? { x: sumX / count, y: sumY / count, z: sumZ / count } : null;
}

function getMinMaxBoundingBox(landmarks, indices) {
    if (!landmarks || indices.length === 0) return null;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    let count = 0;
    for (const index of indices) {
        if (landmarks[index]) {
            minX = Math.min(minX, landmarks[index].x);
            maxX = Math.max(maxX, landmarks[index].x);
            minY = Math.min(minY, landmarks[index].y);
            maxY = Math.max(maxY, landmarks[index].y);
            count++;
        }
    }
    return count > 0 ? { minX, maxX, minY, maxY } : null;
}

function getRawRelativeIrisPosition(faceLandmarks) {
    if (!faceLandmarks || faceLandmarks.length === 0) return null;

    const leftIrisIndices = getUniqueLandmarkIndices(FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS);
    const rightIrisIndices = getUniqueLandmarkIndices(FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS);
    const leftEyeOutlineIndices = getUniqueLandmarkIndices(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE);
    const rightEyeOutlineIndices = getUniqueLandmarkIndices(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE);

    const leftIrisCenter = getAveragePosition(faceLandmarks, leftIrisIndices);
    const rightIrisCenter = getAveragePosition(faceLandmarks, rightIrisIndices);
    const leftEyeBox = getMinMaxBoundingBox(faceLandmarks, leftEyeOutlineIndices);
    const rightEyeBox = getMinMaxBoundingBox(faceLandmarks, rightEyeOutlineIndices);

    let relativeIrisX = [];
    let relativeIrisY = [];

    if (leftIrisCenter && leftEyeBox) {
        const eyeWidth = leftEyeBox.maxX - leftEyeBox.minX;
        const eyeHeight = leftEyeBox.maxY - leftEyeBox.minY;
        if (eyeWidth > 0.001 && eyeHeight > 0.001) {
            relativeIrisX.push((leftIrisCenter.x - leftEyeBox.minX) / eyeWidth);
            relativeIrisY.push((leftIrisCenter.y - leftEyeBox.minY) / eyeHeight);
        }
    }
    if (rightIrisCenter && rightEyeBox) {
        const eyeWidth = rightEyeBox.maxX - rightEyeBox.minX;
        const eyeHeight = rightEyeBox.maxY - rightEyeBox.minY;
        if (eyeWidth > 0.001 && eyeHeight > 0.001) {
            relativeIrisX.push((rightIrisCenter.x - rightEyeBox.minX) / eyeWidth);
            relativeIrisY.push((rightIrisCenter.y - rightEyeBox.minY) / eyeHeight);
        }
    }

    if (relativeIrisX.length === 0) return null;
    const avgRelX = relativeIrisX.reduce((a, b) => a + b, 0) / relativeIrisX.length;
    const avgRelY = relativeIrisY.reduce((a, b) => a + b, 0) / relativeIrisY.length;
    return { avgRelX, avgRelY };
}


function calculateGaze(faceLandmarks) {
    if (isCalibrating && calibrationState !== 'collecting_samples') {
        gazePointElement.style.display = "none"; // Hide gaze point during calibration point transition
        return;
    }

    const rawIrisPos = getRawRelativeIrisPosition(faceLandmarks);
    if (!rawIrisPos) {
        gazePointElement.style.display = "none";
        return;
    }

    if (isCalibrating && calibrationState === 'collecting_samples') {
        accumulatedRelX += rawIrisPos.avgRelX;
        accumulatedRelY += rawIrisPos.avgRelY;
        samplesCollectedThisPoint++;
        
        // Visual feedback for sample collection
        const flashColor = samplesCollectedThisPoint % 10 < 5 ? "lime" : "red";
        calibrationPointElement.style.backgroundColor = flashColor;


        if (samplesCollectedThisPoint >= CALIBRATION_SAMPLES_PER_POINT) {
            const avgCalibRelX = accumulatedRelX / samplesCollectedThisPoint;
            const avgCalibRelY = accumulatedRelY / samplesCollectedThisPoint;
            calibrationDataEye.push({ avgRelX: avgCalibRelX, avgRelY: avgCalibRelY });
            
            currentCalibrationPointIndex++;
            if (currentCalibrationPointIndex < CALIBRATION_POINT_SCREEN_COORDS.length) {
                setupNextCalibrationPoint();
            } else {
                finalizeCalibration();
            }
        }
        return; // Don't draw main gaze point while collecting calibration samples
    }
    
    // If not calibrating, or if calibration is done, use calibrated params
    const { avgRelX, avgRelY } = rawIrisPos;
    const params = calibratedGazeParams;

    let normEyeX = (avgRelX - params.minRelX) / (params.maxRelX - params.minRelX);
    let normEyeY = (avgRelY - params.minRelY) / (params.maxRelY - params.minRelY);

    normEyeX = Math.max(0, Math.min(1, normEyeX));
    normEyeY = Math.max(0, Math.min(1, normEyeY));

    let screenNormX = params.invertX ? (1.0 - normEyeX) : normEyeX;
    let screenNormY = params.invertY ? (1.0 - normEyeY) : normEyeY;
    
    const targetGazeX = screenNormX * window.innerWidth;
    const targetGazeY = screenNormY * window.innerHeight;

    smoothedGazeX = SMOOTHING_FACTOR * targetGazeX + (1 - SMOOTHING_FACTOR) * smoothedGazeX;
    smoothedGazeY = SMOOTHING_FACTOR * targetGazeY + (1 - SMOOTHING_FACTOR) * smoothedGazeY;

    gazePointElement.style.left = `${smoothedGazeX}px`;
    gazePointElement.style.top = `${smoothedGazeY}px`;
    gazePointElement.style.display = "block";
}

function startCalibration() {
    if (!webcamRunning) {
        alert("Please enable the webcam first.");
        return;
    }
    isCalibrating = true;
    calibrationDataEye = [];
    currentCalibrationPointIndex = 0;
    gazePointElement.style.display = "none"; // Hide main gaze point
    calibrationInstructionElement.style.display = "block";
    calibrateButton.innerText = "CALIBRATING...";
    calibrateButton.disabled = true;
    enableWebcamButton.disabled = true;
    setupNextCalibrationPoint();
}

function setupNextCalibrationPoint() {
    calibrationState = "waiting_for_input";
    samplesCollectedThisPoint = 0;
    accumulatedRelX = 0;
    accumulatedRelY = 0;
    
    const point = CALIBRATION_POINT_SCREEN_COORDS[currentCalibrationPointIndex];
    calibrationPointElement.style.left = `${point.x * 100}%`;
    calibrationPointElement.style.top = `${point.y * 100}%`;
    calibrationPointElement.style.backgroundColor = "red";
    calibrationPointElement.style.display = "block";
    calibrationInstructionElement.innerHTML = `Look at the red dot (${currentCalibrationPointIndex + 1}/${CALIBRATION_POINT_SCREEN_COORDS.length})<br>and press SPACEBAR to capture.`;
}

function handleKeyPress(event) {
    if (event.code === "Space" && isCalibrating && calibrationState === "waiting_for_input") {
        event.preventDefault();
        calibrationState = "collecting_samples";
        calibrationInstructionElement.innerHTML = `Capturing... Keep looking at the dot.`;
        // Sample collection happens in predictWebcam via calculateGaze
    }
}

function finalizeCalibration() {
    isCalibrating = false;
    calibrationState = "idle";
    calibrationPointElement.style.display = "none";
    calibrationInstructionElement.style.display = "none";
    gazePointElement.style.display = "block"; // Show main gaze point again
    calibrateButton.innerText = "RECALIBRATE";
    calibrateButton.disabled = false;
    enableWebcamButton.disabled = false;

    if (calibrationDataEye.length !== CALIBRATION_POINT_SCREEN_COORDS.length) {
        console.error("Calibration data mismatch. Using defaults.");
        alert("Calibration failed. Please try again. Using default settings.");
        // Reset to defaults if something went wrong
        calibratedGazeParams = { minRelX: 0.35, maxRelX: 0.65, minRelY: 0.35, maxRelY: 0.65, invertX: true, invertY: false };
        saveCalibrationData();
        return;
    }

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    calibrationDataEye.forEach(p => {
        minX = Math.min(minX, p.avgRelX);
        maxX = Math.max(maxX, p.avgRelX);
        minY = Math.min(minY, p.avgRelY);
        maxY = Math.max(maxY, p.avgRelY);
    });
    
    calibratedGazeParams.minRelX = minX;
    calibratedGazeParams.maxRelX = maxX;
    calibratedGazeParams.minRelY = minY;
    calibratedGazeParams.maxRelY = maxY;

    // Auto-detect inversion based on collected points
    // Points order: TL, TR, BR, BL
    const avgLeftScreenEyeX = (calibrationDataEye[0].avgRelX + calibrationDataEye[3].avgRelX) / 2;
    const avgRightScreenEyeX = (calibrationDataEye[1].avgRelX + calibrationDataEye[2].avgRelX) / 2;
    // If relative X is larger when looking left on screen, then X mapping is inverted
    // (e.g. video mirrored, pupil moves right in image for left screen gaze)
    calibratedGazeParams.invertX = avgLeftScreenEyeX > avgRightScreenEyeX;

    const avgTopScreenEyeY = (calibrationDataEye[0].avgRelY + calibrationDataEye[1].avgRelY) / 2;
    const avgBottomScreenEyeY = (calibrationDataEye[2].avgRelY + calibrationDataEye[3].avgRelY) / 2;
    // If relative Y is larger when looking top of screen, then Y mapping is inverted
    // (e.g. Y landmark decreases upwards, so larger relY for top of screen)
    calibratedGazeParams.invertY = avgTopScreenEyeY > avgBottomScreenEyeY;

    console.log("Calibration complete. New params:", calibratedGazeParams);
    alert("Calibration Complete!");
    saveCalibrationData();
}

function stopCalibration() {
    isCalibrating = false;
    calibrationState = "idle";
    calibrationPointElement.style.display = "none";
    calibrationInstructionElement.style.display = "none";
    if(calibrateButton) {
        calibrateButton.innerText = "START CALIBRATION";
        calibrateButton.disabled = false;
    }
    if(enableWebcamButton) enableWebcamButton.disabled = false;
}

function saveCalibrationData() {
    localStorage.setItem("faceGazeCalibrationParams", JSON.stringify(calibratedGazeParams));
    console.log("Calibration data saved.");
}

function loadCalibrationData() {
    const storedData = localStorage.getItem("faceGazeCalibrationParams");
    if (storedData) {
        calibratedGazeParams = JSON.parse(storedData);
        console.log("Loaded calibration data:", calibratedGazeParams);
        if(calibrateButton) calibrateButton.innerText = "RECALIBRATE";
    } else {
        console.log("No saved calibration data found. Using defaults.");
    }
}

function clearCalibration() {
    localStorage.removeItem("faceGazeCalibrationParams");
    // Reset to default values
    calibratedGazeParams = {
        minRelX: 0.35, maxRelX: 0.65, minRelY: 0.35, maxRelY: 0.65, 
        invertX: true, invertY: false
    };
    console.log("Calibration data cleared. Reset to defaults.");
    alert("Calibration data cleared. Using default settings. Recalibrate for best results.");
    if(calibrateButton) calibrateButton.innerText = "START CALIBRATION";
}


let lastVideoTime = -1;
let results = undefined;
const drawingUtils = new DrawingUtils(canvasCtx);

async function predictWebcam() {
    if (!webcamRunning || video.paused || video.ended) {
        if (webcamRunning) window.requestAnimationFrame(predictWebcam); // keep trying if supposed to be running
        return;
    }

    const videoAR = video.videoHeight / video.videoWidth;
    if (isNaN(videoAR) || videoAR === 0) { // Wait for video metadata
        window.requestAnimationFrame(predictWebcam);
        return;
    }
    
    video.style.width = videoWidth + "px";
    video.style.height = videoWidth * videoAR + "px";
    canvasElement.style.width = videoWidth + "px";
    canvasElement.style.height = videoWidth * videoAR + "px";
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;

    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await faceLandmarker.setOptions({ runningMode: "VIDEO" });
    }

    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        results = faceLandmarker.detectForVideo(video, startTimeMs);
    }

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    if (results && results.faceLandmarks) {
        for (const landmarks of results.faceLandmarks) {
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#C0C0C070", lineWidth: 1 });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "#FF3030" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: "#FF3030" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "#30FF30" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: "#30FF30" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "#E0E0E0" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: "#E0E0E0" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, { color: "#FF3030", lineWidth: 2 });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, { color: "#30FF30", lineWidth: 2 });

            calculateGaze(landmarks); // This now handles both calibration data collection and normal gaze calculation
        }
    } else {
        if (!isCalibrating) gazePointElement.style.display = "none"; // Hide gaze point if no face detected
    }

    if (results && results.faceBlendshapes) {
        drawBlendShapes(videoBlendShapes, results.faceBlendshapes);
    }

    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }
}

function drawBlendShapes(el, blendShapes) {
    if (!blendShapes.length || !blendShapes[0].categories) {
        el.innerHTML = '<li>No blendshape data</li>';
        return;
    }
    let htmlMaker = "";
    blendShapes[0].categories.map((shape) => {
        htmlMaker += `
      <li class="blend-shapes-item">
        <span class="blend-shapes-label">${shape.displayName || shape.categoryName}</span>
        <span class="blend-shapes-value" style="width: calc(${+shape.score * 100}% - 120px); min-width: 30px;">${(+shape.score).toFixed(4)}</span>
      </li>
    `;
    });
    el.innerHTML = htmlMaker;
}

window.addEventListener('load', () => {
    smoothedGazeX = window.innerWidth / 2;
    smoothedGazeY = window.innerHeight / 2;
    if (gazePointElement) {
        gazePointElement.style.left = `${smoothedGazeX}px`;
        gazePointElement.style.top = `${smoothedGazeY}px`;
    }
});

window.addEventListener('resize', () => {
    smoothedGazeX = window.innerWidth / 2;
    smoothedGazeY = window.innerHeight / 2;
    if (gazePointElement && gazePointElement.style.display === 'block' && !isCalibrating) {
      gazePointElement.style.left = `${smoothedGazeX}px`;
      gazePointElement.style.top = `${smoothedGazeY}px`;
    }
    // If calibrating, and window resizes, it's best to restart calibration
    if(isCalibrating) {
        alert("Window resized during calibration. Please restart calibration.");
        stopCalibration();
    }
});
