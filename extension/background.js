chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "detectAI",
        title: "Detect if AI-generated",
        contexts: ["selection"],
    });
});

chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "check-ai-image",
        title: "Check if image is AI-generated",
        contexts: ["image"],
    });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
    if (info.menuItemId === "check-ai-image") {
        const imageUrl = info.srcUrl;
        chrome.storage.local.set({ lastProb: null, lastImage: imageUrl }, () => {
            // Open popup immediately
            // chrome.windows.create({
            //     url: "popup.html",
            //     type: "popup",
            //     width: 500,
            //     height: 350
            // });
            chrome.action.openPopup();
        });
        chrome.storage.local.get(["selectedModel"], (result) => {
            console.log("Saved model is:", result.selectedModel);
            if (result.selectedModel === "model1") {
                checkImageOwnModel(imageUrl);
            }
            else if (result.selectedModel === "model2") {
                checkImageOwnModel2(imageUrl);
            }
        });
    }
});
async function checkImageOwnModel(imageUrl) {
    const url = "http://127.0.0.1:5000/predict";
    chrome.storage.local.set({ checking: true });
    const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },

        body: JSON.stringify({ image: imageUrl }),
    });
    // fetch("http://127.0.0.1:5000/predict", {
    //     method: "POST",
    //     headers: { "Content-Type": "application/json" },
    //     body: JSON.stringify({ image_url: "https://example.com/test.jpg" })
    // })
    //     .then(res => res.json())
    //     .then(console.log)
    //     .catch(console.error);

    const data = await res.json();
    chrome.storage.local.set({ checking: false });
    console.log("Result from own model:", data);
    const category = data.prediction.category || "unknown";
    if (category === "Real") {
        const prob = data.prediction.confidence || 0;
        console.log("REAL");
        chrome.runtime.sendMessage({
            prob: prob,
            imageUrl: imageUrl,
            category: category,
        });
        chrome.storage.local.set({
            lastProb: prob,
            lastImage: imageUrl,
            category: category,
        });
        chrome.storage.local.get({ history: [] }, (result) => {
            const history = result.history;
            history.unshift({ url: imageUrl, prob, category, timestamp: Date.now() }); // latest first
            // Optional: limit to last 50 images
            if (history.length > 50) history.pop();
            chrome.storage.local.set({ history });
        });
    } else {
        const prob = data.prediction.confidence || 0;
        console.log("NOT REAL", category);
        chrome.runtime.sendMessage({
            prob: prob,
            imageUrl: imageUrl,
            category: category,
        });
        chrome.storage.local.set({
            lastProb: prob,
            lastImage: imageUrl,
            category: category,
        });
        chrome.storage.local.get({ history: [] }, (result) => {
            const history = result.history;
            history.unshift({ url: imageUrl, prob, category, timestamp: Date.now() }); // latest first
            // Optional: limit to last 50 images
            if (history.length > 50) history.pop();
            chrome.storage.local.set({ history });
        });
    }
}

async function checkImageOwnModel2(imageUrl) {
    const url = "http://127.0.0.1:5173/predict";
    chrome.storage.local.set({ checking: true });
    const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },

        body: JSON.stringify({ image: imageUrl }),
    });
    // fetch("http://127.0.0.1:5000/predict", {
    //     method: "POST",
    //     headers: { "Content-Type": "application/json" },
    //     body: JSON.stringify({ image_url: "https://example.com/test.jpg" })
    // })
    //     .then(res => res.json())
    //     .then(console.log)
    //     .catch(console.error);

    const data = await res.json();
    chrome.storage.local.set({ checking: false });
    console.log("Result from own model:", data);
    const category = data.prediction.category || "unknown";
    if (category === "Real") {
        const prob = data.prediction.confidence || 0;
        console.log("REAL");
        chrome.runtime.sendMessage({
            prob: 100 - prob,
            imageUrl: imageUrl,
            category: category,
        });
        chrome.storage.local.set({
            lastProb: prob,
            lastImage: imageUrl,
            category: category,
        });
        chrome.storage.local.get({ history: [] }, (result) => {
            const history = result.history;
            history.unshift({ url: imageUrl, prob, category, timestamp: Date.now() }); // latest first
            // Optional: limit to last 50 images
            if (history.length > 50) history.pop();
            chrome.storage.local.set({ history });
        });
    } else {
        const prob = data.prediction.confidence || 0;
        console.log("NOT REAL", category);
        chrome.runtime.sendMessage({
            prob: prob,
            imageUrl: imageUrl,
            category: category,
        });
        chrome.storage.local.set({
            lastProb: prob,
            lastImage: imageUrl,
            category: category,
        });
        chrome.storage.local.get({ history: [] }, (result) => {
            const history = result.history;
            history.unshift({ url: imageUrl, prob, category, timestamp: Date.now() }); // latest first
            // Optional: limit to last 50 images
            if (history.length > 50) history.pop();
            chrome.storage.local.set({ history });
        });
    }
    // const prob = data.prediction.confidence || 0; x``
    // console.log("Probability:", prob);
    // chrome.runtime.sendMessage({ prob: prob, imageUrl: imageUrl });
    // console.log("message sent from background:", prob);

    // //Save probability and image URL to storage
    // chrome.storage.local.set({ lastProb: prob, lastImage: imageUrl });
    // chrome.storage.local.get({ history: [] }, (result) => {
    //     const history = result.history;
    //     history.unshift({ url: imageUrl, prob, timestamp: Date.now() }); // latest first
    //     // Optional: limit to last 50 images
    //     if (history.length > 50) history.pop();
    //     chrome.storage.local.set({ history });
    // });
    // chrome.notifications.create({
    //     type: "basic",
    //     iconUrl: "icon.png",
    //     title: "AI Image Detection",
    //     message: (data.success == true) ? `AI-generated likelihood: ${(prob).toFixed(2)}%` : "Failed"
    // });
}
async function checkImage(imageUrl) {
    console.log("Checking image:", imageUrl);
    const api_user = "1495048728"; // your user ID
    const api_secret = "2r5uQaf9Feekjh4G6W4XtiUex4c8V3QV";

    const url = new URL("https://api.sightengine.com/1.0/check.json");
    url.search = new URLSearchParams({
        models: "genai",
        api_user,
        api_secret,
        url: imageUrl,
    }).toString();

    try {
        chrome.storage.local.set({ checking: true }); // indicate checking
        const res = await fetch(url);
        const data = await res.json();
        console.log("Result:", data);
        chrome.storage.local.set({ checking: false }); // done checking

        const prob = data?.type?.ai_generated || 0;
        chrome.runtime.sendMessage({ prob: prob, imageUrl: imageUrl });

        // Save probability and image URL to storage
        // chrome.storage.local.set({ lastProb: prob, lastImage: imageUrl });
        chrome.storage.local.get({ history: [] }, (result) => {
            const history = result.history;
            history.unshift({ url: imageUrl, prob, timestamp: Date.now() }); // latest first
            // Optional: limit to last 50 images
            if (history.length > 50) history.pop();
            chrome.storage.local.set({ history });
        });

        // Open a small popup window to display result
        // chrome.windows.create({
        //     url: "popup.html",
        //     type: "popup",
        //     width: 500,
        //     height: 350
        // });

        // Optional notification
        chrome.notifications.create({
            type: "basic",
            iconUrl: "icon.png",
            title: "AI Image Detection",
            message:
                data.status == "success"
                    ? `AI-generated likelihood: ${(prob * 100).toFixed(2)}%`
                    : "Failed",
        });
    } catch (err) {
        console.error(err);
        chrome.notifications.create({
            type: "basic",
            iconUrl: "icon.png",
            title: "AI Image Detection",
            message: "Error checking image",
        });
    }
}

const HF_API_URL =
    "https://router.huggingface.co/hf-inference/models/openai-community/roberta-base-openai-detector";
const HF_TOKEN = "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"; // temporary for demo

async function detectAIText(text) {
    const response = await fetch(HF_API_URL, {
        method: "POST",
        headers: {
            Authorization: `Bearer ${HF_TOKEN}`,
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ inputs: text }),
    });
    const data = await response.json();
    return data;
}

// Example usage:
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
    if (info.menuItemId === "detectAI") {
        const selectedText = info.selectionText; // <-- get selected text directly
        if (!selectedText || !selectedText.trim()) {
            chrome.notifications.create({
                type: "basic",
                iconUrl: "icon.png",
                title: "AI Text Detection",
                message: "No text selected!",
            });
            return;
        }

        // call Hugging Face API
        const result = await detectAIText(selectedText);
        console.log("AI detection result:", result);

        const { label, score } = result[0][0] || { label: "Unknown", score: 0 };
        console.log(`Result: ${label} (${score}%)`);
        chrome.notifications.create({
            type: "basic",
            iconUrl: "icon.png",
            title: "AI Text Detection",
            message: `${label} (${(score * 100).toFixed(2)}%)`,
        });
    }
});
