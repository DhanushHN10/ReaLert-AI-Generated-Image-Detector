// === ELEMENTS ===
const idleState = document.getElementById("idle-state");
const imageCardContainer = document.getElementById("image-card-container");
const analyzedImage = document.getElementById("analyzed-image");
const loaderOverlay = document.getElementById("loader-overlay");
const resultContent = document.getElementById("result-content");
const resultBadge = document.getElementById("result-badge");
const confidenceBlock = document.getElementById("confidence-block");
const resultReason = document.getElementById("result-reason");
const errorMessage = document.getElementById("error-message");
const confidenceBar = document.getElementById("confidence-bar");
const confidenceScore = document.getElementById("confidence-score");

// History elements (now in HTML)
const historyBtn = document.getElementById("historyBtn");
const backBtn = document.getElementById("backBtn");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");
const historyContainer = document.getElementById("historyContainer");

// === STATE FUNCTIONS ===
function showIdleState() {
    idleState.style.display = "block";
    imageCardContainer.style.display = "none";
    errorMessage.style.display = "none";
    historyContainer.style.display = "none";
    backBtn.style.display = "none";
    clearHistoryBtn.style.display = "none";
}

function showAnalyzingState() {
    idleState.style.display = "none";
    imageCardContainer.style.display = "block";
    loaderOverlay.style.display = "flex";
    resultContent.style.display = "none";
    errorMessage.style.display = "none";
    historyContainer.style.display = "none";
}

function showResult(prob, imageUrl, category) {
    idleState.style.display = "none";
    historyContainer.style.display = "none";
    imageCardContainer.style.display = "block";
    loaderOverlay.style.display = "none";
    resultContent.style.display = "flex";
    confidenceBlock.style.display = "flex";

    analyzedImage.src = imageUrl;

    const percent = Math.round(prob);
    document.getElementById("category-block").style.display = "flex";
    document.getElementById("category-input").value = category || "Unknown";
    console.log("-------------Category:", category);
    console.log("-------------Showing result for prob:", prob);

    // confidenceBar.style.width = `${percent}%`;
    // confidenceBar.style.height = "16px";
    // confidenceBar.style.borderRadius = "8px";
    // confidenceBar.style.transition = "width 0.5s ease-in-out";
    //confidenceBar.textContent = `${percent}%`;
    confidenceBar.style.width = `${percent}%`;
    confidenceBar.style.backgroundColor = "#4caf50"; // default green
    confidenceScore.textContent = `${percent}%`;
    console.log("Confidence percent:", percent);
    if (category == "Real") {
        prob = 100 - prob;
    }

    // Set badge based on probability
    if (prob > 70) {
        resultBadge.textContent = "Likely AI Generated 🤖";
        resultBadge.style.color = "#e53935";
        confidenceBar.style.backgroundColor = "#e53935"; // red
        console.log("Set confidence bar to red");
    } else if (prob > 40) {
        resultBadge.textContent = "Possibly AI Generated ⚠️";
        resultBadge.style.color = "#fbc02d";
        confidenceBar.style.backgroundColor = "#fbc02d"; // yellow
    } else {
        resultBadge.textContent = "Likely Real 🖼️";
        resultBadge.style.color = "#4caf50";
        confidenceBar.style.backgroundColor = "#4caf50"; // green
    }

    resultReason.textContent = "";
}


const modelSelect = document.getElementById("modelSelect");

// Load the last selected model when popup opens
chrome.storage.local.get("selectedModel", (data) => {
    if (data.selectedModel) {
        modelSelect.value = data.selectedModel;
    }
});
function showError(message = "Something went wrong while analyzing the image.") {
    idleState.style.display = "none";
    imageCardContainer.style.display = "none";
    historyContainer.style.display = "none";
    errorMessage.style.display = "block";
    errorMessage.textContent = message;
}

// === HISTORY FUNCTIONS ===
function loadHistory() {
    chrome.storage.local.get({ history: [] }, (data) => {
        const history = data.history;
        historyContainer.innerHTML = "";
        if (history.length === 0) {
            const msg = document.createElement("p");
            msg.textContent = "No images scanned yet.";
            msg.style.color = "#9ca3af";
            historyContainer.appendChild(msg);
            return;
        }

        history.forEach(item => {
            const div = document.createElement("div");
            div.className = "history-item";
            div.innerHTML = `
                <img src="${item.url}" width="50" height="50" />
                <span>${(item.category)}</span>

                <span>${(item.prob).toFixed(2)}%</span>

            `;
            div.style.display = "flex";
            div.style.alignItems = "center";
            div.style.cursor = "pointer";
            div.style.marginBottom = "0.5rem";

            div.addEventListener("click", () => {
                showResult(item.prob, item.url, item.category);
                backBtn.style.display = "none";
                clearHistoryBtn.style.display = "none";
            });

            historyContainer.appendChild(div);
        });
    });
}

// === EVENT LISTENERS ===
document.addEventListener("DOMContentLoaded", () => {

    // Load last analyzed image
    const modelBtn = document.getElementById("modelBtn");
    const modelDropdown = document.getElementById("modelDropdown");

    modelBtn.addEventListener("click", () => {
        modelDropdown.classList.toggle("hidden");
    });
    chrome.storage.local.get(["lastProb", "lastImage", "category"], (result) => {
        const prob = result.lastProb;
        const imageUrl = result.lastImage;

        if (!imageUrl) {
            showIdleState();
            return;
        }

        analyzedImage.src = imageUrl;

        if (prob === null || prob === undefined) {
            showAnalyzingState();
        } else {
            console.log("Loaded from storage - prob:", prob, "imageUrl:", imageUrl, "category:", result.category);
            showResult(prob, imageUrl, result.category);
        }
    });

    document.querySelectorAll(".dropdown-option").forEach(option => {
        option.addEventListener("click", () => {
            const selectedModel = option.dataset.model;
            chrome.storage.local.set({ selectedModel });
            modelDropdown.classList.add("hidden");
            console.log("Selected model:", selectedModel);
        });
    });

    // Close dropdown when clicking outside
    document.addEventListener("click", (event) => {
        if (!modelBtn.contains(event.target) && !modelDropdown.contains(event.target)) {
            modelDropdown.classList.add("hidden");
        }
    });
});

// History button
historyBtn.addEventListener("click", () => {
    idleState.style.display = "none";
    imageCardContainer.style.display = "none";
    historyContainer.style.display = "block";
    backBtn.style.display = "inline-block";
    clearHistoryBtn.style.display = "inline-block";
    loadHistory();
});

// Back button
backBtn.addEventListener("click", () => {
    historyContainer.style.display = "none";
    imageCardContainer.style.display = "block";
    backBtn.style.display = "none";
    clearHistoryBtn.style.display = "none";
});

// Clear history
clearHistoryBtn.addEventListener("click", () => {
    chrome.storage.local.remove("history", () => {
        loadHistory();
    });
});

// Message listener
chrome.runtime.onMessage.addListener((msg) => {
    if (msg.prob !== undefined && msg.imageUrl) {
        showResult(msg.prob, msg.imageUrl, msg.category);
    } else if (msg.error) {
        showError(msg.error);
    }
});
modelSelect.addEventListener("change", () => {
    const selectedModel = modelSelect.value;
    chrome.storage.local.set({ selectedModel });
    console.log("Saved model:", selectedModel);
});

