const historyContainer = document.getElementById("historyContainer");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");

function loadHistory() {
    // chrome.storage.local.get("scanHistory", (data) => {
    //     const history = data.scanHistory || [];
    //     if (history.length === 0) {
    //         historyContainer.innerHTML = "<p style='color:#64748b;'>No scans yet.</p>";
    //         return;
    //     }

    //     historyContainer.innerHTML = history.map((item) => `
    //   <div class="history-item">
    //     <img src="${item.image}" />
    //     <span>
    //       <strong>${item.result}</strong> (${item.probability}%)
    //     </span>
    //   </div>
    // `).join("");
    // });
    const container = document.getElementById('historyContainer');
    const clearBtn = document.getElementById('clearHistoryBtn');

    chrome.storage.local.get({ history: [] }, (data) => {
        const history = data.history;
        if (history.length === 0) {
            container.textContent = 'No images scanned yet.';
            return;
        }

        history.forEach(item => {
            const div = document.createElement('div');
            div.className = 'historyItem';
            div.innerHTML = `
                <img src="${item.url}" width="50" height="50">
                <span>${(item.prob * 100).toFixed(2)}%</span>
            `;
            container.appendChild(div);
        });
    });
    clearBtn.addEventListener('click', () => {
        chrome.storage.local.remove('history', () => {
            renderHistory(); // re-render after clearing
            console.log('History cleared!');
        });
    });
}

clearHistoryBtn.addEventListener("click", () => {
    chrome.storage.local.remove("scanHistory", loadHistory);
});

loadHistory();
